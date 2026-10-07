"""知识库的分类汇总。

仪表盘原先只在视图里内联了四个 ``count()``，看不出「这批论文是什么构成」——
不知道年份分布、不知道关键词集中在哪些方向、不知道有多少篇还没读。
要做分类汇总就必须有按值聚合的查询，那些查询放在视图里不合适（别处也要用），
所以收拢到这一层。

**每个维度都补齐了空值。** 例如阅读状态有四种，库里可能一种都没有（全 unread），
如果只 ``GROUP BY`` 出实际存在的值，前端就得自己猜「缺的那个是 0 还是不存在」，
图表的比例也会算错。这里统一按**定义顺序**输出全部取值，缺的填 0。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func

from ..extensions import db
from ..models import Chunk, CodeRepo, Note, Paper, Tag
from ..models.note import KINDS as NOTE_KINDS
from ..models.paper import INGEST_STATUSES, READING_STATUSES

# 各维度的显示名。放在这里而不是模板里：同一个维度可能在多处渲染，
# 名字写两遍迟早会出现「待读」和「未读」两种说法并存。
READING_LABELS = {
    "unread": "待读",
    "reading": "在读",
    "read": "已读",
    "reviewed": "已复看",
}
INGEST_LABELS = {
    "pending": "待解析",
    "parsed": "已解析",
    "indexed": "已索引",
    "failed": "失败",
}
NOTE_KIND_LABELS = {
    "summary": "摘要",
    "deep_read": "精读",
    "code_review": "代码对照",
    "qa": "问答",
    "insight": "想法",
    "manual": "手写",
}
DIMENSION_LABELS = {
    "topic": "主题",
    "method": "方法",
    "task": "任务",
    "domain": "领域",
    "venue": "会议",
    "status": "状态",
    "misc": "其它",
}


def _live_papers():
    """未软删除的论文。所有统计都基于这个集合——
    把已移出文库的论文算进「年份分布」会让数字对不上文库页的篇数。"""
    return db.session.query(Paper).filter(Paper.deleted_at.is_(None))


def _grouped(column, *, limit: int | None = None, exclude_empty: bool = True) -> list[tuple]:
    query = (
        db.session.query(column, func.count(Paper.id))
        .filter(Paper.deleted_at.is_(None))
        .group_by(column)
        .order_by(func.count(Paper.id).desc())
    )
    if exclude_empty:
        query = query.filter(column.isnot(None), column != "")
    if limit:
        query = query.limit(limit)
    return query.all()


def overview() -> dict[str, Any]:
    """顶部数字卡。

    注意 ``reviewed`` 与 ``indexed`` 两项：旧版 ``papers.stats()`` 把它们漏了，
    于是四种阅读状态加起来不等于总数——分类汇总页上这种对不上会立刻被看见。
    """
    total = _live_papers().count()
    by_reading = dict(_grouped(Paper.reading_status))
    by_ingest = dict(_grouped(Paper.ingest_status))

    pages = (
        db.session.query(func.coalesce(func.sum(Paper.page_count), 0))
        .filter(Paper.deleted_at.is_(None))
        .scalar()
    )
    tokens = db.session.query(func.coalesce(func.sum(Chunk.n_tokens), 0)).scalar()

    return {
        "papers": total,
        "notes": db.session.query(Note).count(),
        "chunks": db.session.query(Chunk).count(),
        "tags": db.session.query(Tag).count(),
        "pages": int(pages or 0),
        "tokens": int(tokens or 0),
        "reading": {key: by_reading.get(key, 0) for key in READING_STATUSES},
        "ingest": {key: by_ingest.get(key, 0) for key in INGEST_STATUSES},
    }


def by_year() -> list[dict[str, Any]]:
    rows = _grouped(Paper.year)
    # 年份升序更适合看趋势，而 _grouped 是按数量排的
    return [{"label": str(year), "value": count} for year, count in sorted(rows)]


def by_venue(limit: int = 12) -> list[dict[str, Any]]:
    rows = _grouped(Paper.venue, limit=limit)
    return [{"label": venue, "value": count} for venue, count in rows]


def by_reading_status() -> list[dict[str, Any]]:
    counts = dict(_grouped(Paper.reading_status))
    return [
        {"key": key, "label": READING_LABELS.get(key, key), "value": counts.get(key, 0)}
        for key in READING_STATUSES
    ]


def by_ingest_status() -> list[dict[str, Any]]:
    counts = dict(_grouped(Paper.ingest_status))
    return [
        {"key": key, "label": INGEST_LABELS.get(key, key), "value": counts.get(key, 0)}
        for key in INGEST_STATUSES
    ]


def by_note_kind() -> list[dict[str, Any]]:
    rows = (
        db.session.query(Note.kind, func.count(Note.id))
        .group_by(Note.kind)
        .order_by(func.count(Note.id).desc())
        .all()
    )
    counts = dict(rows)
    # 按定义顺序输出，没有的补 0；未在 KINDS 里的（历史数据）附在最后
    ordered = [
        {"key": kind, "label": NOTE_KIND_LABELS.get(kind, kind), "value": counts.pop(kind, 0)}
        for kind in NOTE_KINDS
    ]
    ordered.extend(
        {"key": kind, "label": kind, "value": count}
        for kind, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    return [item for item in ordered if item["value"] or item["key"] in NOTE_KINDS]


def by_tag_dimension(limit_per_dimension: int = 8) -> dict[str, Any]:
    """按维度分组的标签热度。

    「主题/方法/任务/领域」这些维度上的分布，比「哪个标签用得最多」更有信息量：
    它能回答「我的库主要在研究什么」这个问题。
    """
    # 与 tag_facets 同一口径：论文自身的标签 ∪ 其笔记的标签
    from ..models import NoteTag, PaperTag

    pairs = (
        db.session.query(PaperTag.tag_id.label("tag_id"), PaperTag.paper_id.label("paper_id"))
        .union(
            db.session.query(NoteTag.tag_id, Note.paper_id)
            .join(Note, Note.id == NoteTag.note_id)
            .filter(Note.paper_id.isnot(None))
        )
        .subquery()
    )
    rows = (
        db.session.query(
            Tag.dimension,
            Tag.name,
            func.count(func.distinct(pairs.c.paper_id)),
        )
        .join(pairs, pairs.c.tag_id == Tag.id)
        .join(Paper, Paper.id == pairs.c.paper_id)
        .filter(Paper.deleted_at.is_(None))
        .group_by(Tag.id)
        .order_by(func.count(func.distinct(pairs.c.paper_id)).desc())
        .all()
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for dimension, name, count in rows:
        bucket = grouped.setdefault(dimension or "misc", [])
        if len(bucket) < limit_per_dimension:
            bucket.append({"label": name, "value": count})
    return {
        DIMENSION_LABELS.get(dim, dim): items
        for dim, items in sorted(grouped.items(), key=lambda kv: -sum(i["value"] for i in kv[1]))
        if items
    }


def code_coverage() -> dict[str, int]:
    """有多少篇论文关联了开源代码。

    用 distinct 而不是 count(*)：一篇论文可能关联多个仓库
    （官方实现 + 第三方复现），直接数行会把有代码的论文数算多。
    """
    with_code = (
        db.session.query(func.count(func.distinct(CodeRepo.paper_id)))
        .filter(CodeRepo.paper_id.isnot(None))
        .scalar()
    ) or 0
    total = _live_papers().count()
    return {"with_code": int(with_code), "without_code": max(0, total - int(with_code))}


def chunk_kinds() -> list[dict[str, Any]]:
    rows = (
        db.session.query(Chunk.kind, func.count(Chunk.id))
        .group_by(Chunk.kind)
        .order_by(func.count(Chunk.id).desc())
        .all()
    )
    return [{"label": kind or "未知", "value": count} for kind, count in rows]


def collect() -> dict[str, Any]:
    """一次性取回分类汇总页需要的全部数据。"""
    return {
        "overview": overview(),
        "years": by_year(),
        "venues": by_venue(),
        "reading": by_reading_status(),
        "ingest": by_ingest_status(),
        "note_kinds": by_note_kind(),
        "tags": by_tag_dimension(),
        "code": code_coverage(),
        "chunks": chunk_kinds(),
    }


__all__ = [
    "DIMENSION_LABELS",
    "INGEST_LABELS",
    "NOTE_KIND_LABELS",
    "READING_LABELS",
    "by_ingest_status",
    "by_note_kind",
    "by_reading_status",
    "by_tag_dimension",
    "by_venue",
    "by_year",
    "chunk_kinds",
    "code_coverage",
    "collect",
    "overview",
]
