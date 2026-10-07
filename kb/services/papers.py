"""论文的增删改查。

网页端与对外接口共用这一层。两边的差别只在「谁能做什么」（权限），
而不在「怎么做」——同一份业务规则实现两遍，迟早会分叉。

论文的删除一律是**软删除**：磁盘上的文件和数据库里的记录都保留，
只标记 ``deleted_at``。论文记录是笔记、标签、阅读进度的挂载点，
真删了会连带毁掉用户自己写的东西。要彻底清除得显式调用 ``purge``。
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import or_

from ..extensions import db
from ..models import Chunk, Note, Paper, PaperDuplicate
from ..models.base import utcnow
from ..models.paper import (
    INGEST_FAILED,
    INGEST_PARSED,
    INGEST_PENDING,
    READING_STATUSES,
    SOURCE_UPLOAD,
    SOURCE_URL,
)
from .paths import sanitize_filename

log = logging.getLogger(__name__)


class PaperError(ValueError):
    """论文操作失败。消息面向用户。"""


# --------------------------------------------------------------------------
# 查询
# --------------------------------------------------------------------------

_SORTS = {
    "added": Paper.created_at.desc(),
    "added_asc": Paper.created_at.asc(),
    "title": Paper.title.asc(),
    "year": Paper.year.desc(),
    "year_asc": Paper.year.asc(),
    "updated": Paper.updated_at.desc(),
}

# 排序选项的**唯一来源**：界面下拉框和视图校验都从这里取。
# 之前视图和模板各写了一份列表，结果是视图支持 added_asc/updated
# 但界面上根本选不到——两边各改各的，迟早对不上。
SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("added", "最新入库"),
    ("added_asc", "最早入库"),
    ("year", "年份新→旧"),
    ("year_asc", "年份旧→新"),
    ("title", "按标题"),
    ("updated", "最近更新"),
)
SORT_VALUES = frozenset(value for value, _ in SORT_OPTIONS)


def _paper_ids_for_tag(tag_id: str):
    """「具备某个标签」的论文 ID 子查询。

    **标签主要挂在笔记上，不在论文上。** AI 精读产出的是 note_tags，
    而 paper_tags 到现在一行都没写过。所以只查 paper_tags 的话，
    按标签筛选会永远返回 0 篇，而侧栏的计数又来自 Tag.usage_count（显示 59）——
    「显示 59 篇、点进去 0 篇」这种错位比干脆没有筛选功能更让人困惑。

    因此定义成两条路径的并集：
      * 论文自己挂了该标签；
      * 论文的任意一篇笔记挂了该标签。
    计数与筛选必须走同一个定义，否则数字和结果对不上。
    """
    from ..models import NoteTag, PaperTag

    by_paper = db.session.query(PaperTag.paper_id).filter(PaperTag.tag_id == tag_id)
    by_note = (
        db.session.query(Note.paper_id)
        .join(NoteTag, NoteTag.note_id == Note.id)
        .filter(NoteTag.tag_id == tag_id, Note.paper_id.isnot(None))
    )
    return by_paper.union(by_note)


def list_papers(
    *,
    query: str | None = None,
    tag_ids: list[str] | None = None,
    year: int | None = None,
    venue: str | None = None,
    reading_status: str | None = None,
    ingest_status: str | None = None,
    has_repo: bool | None = None,
    include_deleted: bool = False,
    sort: str = "added",
    limit: int = 50,
    cursor: str | None = None,
) -> tuple[list[Paper], str | None, int]:
    """按条件查询论文。返回 ``(结果, 下一页游标, 总数)``。

    游标用 ULID 主键：它本身按时间有序，所以「取比游标更早的记录」
    就是一句 ``id < cursor``，而且翻页期间新入库的论文不会导致漏条或重复
    （用 offset 就会）。
    """

    base = db.session.query(Paper)

    if not include_deleted:
        base = base.filter(Paper.deleted_at.is_(None))

    if query:
        # 标题、作者、摘要、标识符都搜。
        # 这里用 LIKE 而不是 FTS5：论文只有几百到几万条，LIKE 足够快，
        # 而 FTS5 索引是为「分块」量级（大两个数量级）准备的。
        pattern = f"%{query.strip()}%"
        base = base.filter(
            or_(
                Paper.title.ilike(pattern),
                Paper.abstract.ilike(pattern),
                Paper.doi.ilike(pattern),
                Paper.arxiv_id.ilike(pattern),
                Paper.venue.ilike(pattern),
            )
        )

    if year:
        base = base.filter(Paper.year == year)
    if venue:
        base = base.filter(Paper.venue.ilike(f"%{venue}%"))
    if reading_status and reading_status in READING_STATUSES:
        base = base.filter(Paper.reading_status == reading_status)
    if ingest_status:
        base = base.filter(Paper.ingest_status == ingest_status)

    if tag_ids:
        # 要求同时具备所有指定标签（AND 语义），这是筛选时更符合直觉的行为
        for tag_id in tag_ids:
            base = base.filter(Paper.id.in_(_paper_ids_for_tag(tag_id)))

    if has_repo is not None:
        from ..models import CodeRepo

        repo_paper_ids = db.session.query(CodeRepo.paper_id).filter(CodeRepo.paper_id.isnot(None))
        if has_repo:
            base = base.filter(Paper.id.in_(repo_paper_ids))
        else:
            base = base.filter(Paper.id.notin_(repo_paper_ids))

    total = base.count()

    order = _SORTS.get(sort, _SORTS["added"])
    ordered = base.order_by(Paper.id.desc()) if sort in {"added", "updated"} else base.order_by(order)

    if cursor:
        ordered = ordered.filter(Paper.id < cursor)

    rows = ordered.limit(limit + 1).all()
    next_cursor = rows[limit].id if len(rows) > limit else None
    return rows[:limit], next_cursor, total


def get_paper(paper_id: str, *, include_deleted: bool = False) -> Paper | None:
    paper = db.session.get(Paper, paper_id)
    if paper is None:
        return None
    if paper.deleted_at is not None and not include_deleted:
        return None
    return paper


def get_paper_by_path(path: str) -> Paper | None:
    from .paths import path_key

    return db.session.query(Paper).filter(Paper.path_key == path_key(path)).one_or_none()


def get_or_404(paper_id: str) -> Paper:
    paper = get_paper(paper_id)
    if paper is None:
        raise PaperError("论文不存在")
    return paper


def filter_options() -> dict[str, Any]:
    """筛选器可选项：年份、会议、标签。供界面渲染下拉框。"""
    from ..models import Tag

    years = [
        row[0]
        for row in db.session.query(Paper.year)
        .filter(Paper.deleted_at.is_(None), Paper.year.isnot(None))
        .distinct()
        .order_by(Paper.year.desc())
        .all()
    ]
    venues = [
        row[0]
        for row in db.session.query(Paper.venue)
        .filter(Paper.deleted_at.is_(None), Paper.venue.isnot(None), Paper.venue != "")
        .distinct()
        .order_by(Paper.venue)
        .limit(200)
        .all()
    ]
    tags = db.session.query(Tag).order_by(Tag.dimension, Tag.usage_count.desc()).all()
    return {"years": years, "venues": venues, "tags": tags}


def tag_facets() -> dict[str, list[dict[str, Any]]]:
    """按维度分组的标签分面，每个标签带**论文数**。

    计数口径必须与 ``list_papers(tag_ids=...)`` 完全一致，否则会出现
    「徽章写着 59 篇、点进去 0 篇」。所以这里复用同一套「哪些论文算具备该标签」
    的定义（见 ``_paper_ids_for_tag``），而不是数关联表行数，也不是用
    ``Tag.usage_count`` 那个冗余计数器——后者把已移出文库的论文也算进去了。

    计数用一条 GROUP BY 出，不做逐个标签的 ``count()``：那是 N+1 查询，
    标签上百时肉眼可见地慢。
    """
    from ..models import NoteTag, PaperTag, Tag

    # (tag_id, paper_id) 的全部关联，两条来源合并后去重计数
    pairs = (
        db.session.query(
            PaperTag.tag_id.label("tag_id"), PaperTag.paper_id.label("paper_id")
        )
        .union(
            db.session.query(NoteTag.tag_id, Note.paper_id)
            .join(Note, Note.id == NoteTag.note_id)
            .filter(Note.paper_id.isnot(None))
        )
        .subquery()
    )
    counts = dict(
        db.session.query(
            pairs.c.tag_id,
            db.func.count(db.distinct(pairs.c.paper_id)),
        )
        .join(Paper, Paper.id == pairs.c.paper_id)
        .filter(Paper.deleted_at.is_(None))
        .group_by(pairs.c.tag_id)
        .all()
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for tag in db.session.query(Tag).all():
        grouped.setdefault(tag.dimension or "misc", []).append(
            {
                "id": tag.id,
                "name": tag.name,
                "dimension": tag.dimension,
                "color": tag.color,
                "count": counts.get(tag.id, 0),
            }
        )
    for items in grouped.values():
        items.sort(key=lambda item: (-item["count"], item["name"]))
    return grouped


# --------------------------------------------------------------------------
# 修改
# --------------------------------------------------------------------------

_UPDATABLE = {
    "title", "authors", "year", "venue", "doi", "arxiv_id", "abstract",
    "reading_status", "rating", "language",
}


def update_paper(paper_id: str, values: dict) -> Paper:
    """更新论文的可编辑字段。

    标题改动会同步更新归一化标题——模糊去重依赖它，不同步的话
    用户手改标题之后去重就会失效。
    """
    paper = get_or_404(paper_id)
    changed = []

    for key, value in values.items():
        if key not in _UPDATABLE:
            continue
        if key == "reading_status" and value not in READING_STATUSES:
            raise PaperError(f"无效的阅读状态：{value}")
        if key == "rating" and value not in (None, 0, 1, 2, 3, 4, 5):
            raise PaperError("评分需要在 0-5 之间")
        if getattr(paper, key) != value:
            setattr(paper, key, value)
            changed.append(key)

    if "title" in changed:
        from .pdf import normalize_title

        paper.title_norm = normalize_title(paper.title)
        paper.meta = {**(paper.meta or {}), "title_source": "manual"}

    if changed:
        db.session.commit()
        log.info("更新论文 %s：%s", paper_id, ", ".join(changed))
    return paper


def soft_delete(paper_id: str) -> Paper:
    """软删除。文件与笔记都保留。"""
    paper = get_or_404(paper_id)
    paper.deleted_at = utcnow()
    db.session.commit()
    log.info("软删除论文 %s（%s）", paper_id, paper.title)
    return paper


def restore(paper_id: str) -> Paper:
    paper = db.session.get(Paper, paper_id)
    if paper is None:
        raise PaperError("论文不存在")
    paper.deleted_at = None
    db.session.commit()
    return paper


def purge(paper_id: str) -> None:
    """彻底删除。**不可逆**，笔记会保留但失去论文关联。

    这是唯一会真正丢数据的操作，所以调用点必须显式且经过确认。
    """
    paper = db.session.get(Paper, paper_id)
    if paper is None:
        raise PaperError("论文不存在")

    note_count = db.session.query(Note).filter(Note.paper_id == paper_id).count()
    db.session.delete(paper)
    db.session.commit()
    log.warning("彻底删除论文 %s（%d 篇笔记已解除关联但保留）", paper_id, note_count)


def mark_reading(paper_id: str, status: str) -> Paper:
    return update_paper(paper_id, {"reading_status": status})


# --------------------------------------------------------------------------
# 新建
# --------------------------------------------------------------------------


def create_from_path(path: str, *, source: str = SOURCE_URL, title: str | None = None) -> Paper:
    """把磁盘上的一个 PDF 登记为论文。

    上传与链接入库最终都落到这里——文件先落盘，再走和扫描完全相同的
    登记路径。这样「上传的论文」和「扫描发现的论文」在库里没有任何区别，
    不会出现两套逻辑各自处理一半字段的情况。
    """
    import os

    from .paths import path_key

    if not os.path.isfile(path):
        raise PaperError(f"文件不存在：{path}")

    from .pdf import PdfError, normalize_title, read_metadata

    key = path_key(path)
    existing = db.session.query(Paper).filter(Paper.path_key == key).one_or_none()
    if existing is not None:
        if existing.deleted_at is not None:
            existing.deleted_at = None
            db.session.commit()
        return existing

    stat = os.stat(path)

    # **先在事务外做慢 I/O，再开写事务。**
    #
    # SQLite 同一时刻只允许一个写事务。如果先 add() 拿到写锁、再解析 PDF
    # （可能要几秒），这段时间里所有其它写入者都会阻塞；批量导入时
    # 后台 worker 等满 busy_timeout 就会抛 `database is locked`。
    # 读元数据是只读操作，完全可以放在拿锁之前。
    parsed: dict = {}
    try:
        meta = read_metadata(path)
        parsed = {
            "title": title or meta.title or os.path.basename(path),
            "authors": meta.authors,
            "year": meta.year,
            "doi": meta.doi,
            "arxiv_id": meta.arxiv_id,
            "abstract": meta.abstract,
            "page_count": meta.page_count,
            "meta": {"title_source": meta.title_source},
        }
    except PdfError as exc:
        parsed = {
            "title": title or os.path.basename(path),
            "meta": {"metadata_error": str(exc)},
        }
        log.warning("登记论文时读取元数据失败 %s：%s", path, exc)

    paper = Paper(
        file_path=os.path.abspath(path),
        path_key=key,
        file_size=stat.st_size,
        file_mtime_ns=stat.st_mtime_ns,
        source=source,
        ingest_status=INGEST_PENDING,
    )
    paper.title = parsed.get("title", "")
    paper.title_norm = normalize_title(paper.title)
    paper.authors = parsed.get("authors")
    paper.year = parsed.get("year")
    paper.doi = parsed.get("doi")
    paper.arxiv_id = parsed.get("arxiv_id")
    paper.abstract = parsed.get("abstract")
    paper.page_count = parsed.get("page_count")
    paper.meta = parsed.get("meta") or {}

    db.session.add(paper)
    db.session.commit()
    return paper


def store_upload(file_storage, *, max_mb: int = 100) -> Paper:
    """保存上传的 PDF 并登记为论文。

    安全措施（上传接口是外部输入，必须当作敌意数据处理）：
      * 检查 ``%PDF-`` 魔术字节——只看扩展名等于没有校验；
      * 文件名经过清洗，防止 ``../../etc/passwd`` 之类的路径穿越，
        也防止 Windows 保留名（CON、NUL）导致写入失败；
      * 内容寻址存储（按哈希分目录），重复上传同一份文件不会占两份空间。
    """
    import hashlib
    import os

    from flask import current_app

    from .pdf import looks_like_pdf

    cfg = current_app.extensions["kb_boot_config"]

    original = file_storage.filename or "upload.pdf"
    safe_name = sanitize_filename(original, fallback="upload.pdf")
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    # 先落盘到临时文件，校验通过再挪到内容寻址的最终位置
    tmp_dir = cfg.uploads_dir / "_incoming"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{os.getpid()}-{id(file_storage)}.part"

    digest = hashlib.sha256()
    size = 0
    limit = max_mb * 1024 * 1024

    try:
        with open(tmp_path, "wb") as fh:
            while chunk := file_storage.stream.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise PaperError(f"文件超过上限 {max_mb} MB")
                digest.update(chunk)
                fh.write(chunk)

        if size == 0:
            raise PaperError("上传的文件是空的")

        if not looks_like_pdf(tmp_path):
            raise PaperError("这不是一个有效的 PDF 文件（文件头校验未通过）")

        hexdigest = digest.hexdigest()
        target_dir = cfg.uploads_dir / hexdigest[:2]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{hexdigest[:16]}-{safe_name}"

        if target.exists():
            tmp_path.unlink(missing_ok=True)
        else:
            os.replace(tmp_path, target)

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    paper = create_from_path(str(target), source=SOURCE_UPLOAD)
    log.info("已上传论文：%s（%d KB）", paper.title[:60], size // 1024)
    return paper


# --------------------------------------------------------------------------
# 重复候选处理
# --------------------------------------------------------------------------


def list_duplicates(status: str = "pending") -> list[tuple[PaperDuplicate, Paper, Paper]]:
    """列出重复候选，连同伴随的两条论文记录一起返回。"""
    rows = (
        db.session.query(PaperDuplicate)
        .filter(PaperDuplicate.status == status)
        .order_by(PaperDuplicate.created_at.desc())
        .all()
    )
    out = []
    for row in rows:
        left = db.session.get(Paper, row.paper_id)
        right = db.session.get(Paper, row.candidate_id)
        if left and right:
            out.append((row, left, right))
    return out


def resolve_duplicate(dup_id: str, action: str) -> None:
    """处理重复候选。

    ``keep_left`` / ``keep_right`` 会把另一条移入软删除状态（**不**物理删除，
    这样误操作还能恢复）；``dismiss`` 表示「它们确实是两篇不同的论文」。
    """
    row = db.session.get(PaperDuplicate, dup_id)
    if row is None:
        raise PaperError("重复记录不存在")

    if action == "dismiss":
        row.status = "dismissed"
    elif action in {"keep_left", "keep_right"}:
        keep_id = row.paper_id if action == "keep_left" else row.candidate_id
        drop_id = row.candidate_id if action == "keep_left" else row.paper_id
        keeper = db.session.get(Paper, keep_id)
        dropped = db.session.get(Paper, drop_id)
        if keeper is None or dropped is None:
            raise PaperError("重复记录引用的论文已不存在")

        # 把被丢弃那篇的笔记转移到保留的那篇，避免笔记变成孤儿
        moved_notes = (
            db.session.query(Note).filter(Note.paper_id == drop_id).update({"paper_id": keep_id})
        )
        # 补上保留方缺失的标识符
        for field in ("doi", "arxiv_id", "year", "venue", "abstract"):
            if not getattr(keeper, field) and getattr(dropped, field):
                setattr(keeper, field, getattr(dropped, field))

        dropped.deleted_at = utcnow()
        row.status = "merged"
        log.info(
            "合并重复论文：保留 %s，弃用 %s（转移 %d 篇笔记）",
            keep_id, drop_id, moved_notes,
        )
    else:
        raise PaperError(f"未知的处理方式：{action}")

    row.resolved_at = utcnow()
    row.resolved_by = "web"
    db.session.commit()


def paper_dict(paper: Paper, *, detail: bool = False) -> dict:
    """论文的对外表示。

    列表与详情用同一个序列化函数，靠 ``detail`` 开关控制字段量。
    分成两个函数写的话，两边迟早会漏掉同一批字段的维护。

    **放在服务层而不是 API 层**：MCP 工具、将来的其它接入面都要用同一份
    序列化结果。让服务层去 import API 是把依赖方向搞反了——API 依赖服务，
    不是反过来。所以从这里定义，API 侧 import 它。
    """
    from ..utils.time import iso

    data = {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors or [],
        "year": paper.year,
        "venue": paper.venue,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "source": paper.source,
        "reading_status": paper.reading_status,
        "ingest_status": paper.ingest_status,
        "rating": paper.rating,
        "page_count": paper.page_count,
        "has_file": paper.has_file,
        "created_at": iso(paper.created_at),
        "updated_at": iso(paper.updated_at),
        "deleted_at": iso(paper.deleted_at),
    }
    if detail:
        data.update(
            {
                "abstract": paper.abstract,
                "language": paper.language,
                "file_size": paper.file_size,
                "file_hash": paper.file_hash,
                "error": paper.error,
                "tags": [
                    {
                        "id": link.tag.id,
                        "name": link.tag.name,
                        "dimension": link.tag.dimension,
                        "source": link.source,
                        "confidence": link.confidence,
                    }
                    for link in paper.tag_links
                ],
                "repos": [
                    {
                        "id": repo.id,
                        "name": repo.name,
                        "url": repo.url,
                        "local_path": repo.local_path,
                        "mapping_confidence": repo.mapping_confidence,
                    }
                    for repo in paper.repos
                ],
                "notes": [
                    {
                        "id": note.id,
                        "title": note.title,
                        "kind": note.kind,
                        "status": note.status,
                    }
                    for note in paper.notes
                    if note.paper_id == paper.id
                ],
            }
        )
    return data


def paper_text(
    paper_id: str,
    *,
    page: int | None = None,
    section: str | None = None,
    offset: int = 0,
    limit: int = 12,
) -> dict[str, Any]:
    """取一篇论文的正文分块，带定位信息。

    **这是「检索 → 读原文 → 引用」里中间那一步。** 检索只给回几个片段，
    想回答「这个方法具体怎么做」往往要往下读；没有这个入口，调用方只能
    拿着 chunk_id 干瞪眼。

    三种读法，按 agent 的实际动线设计：

    * 什么都不传——返回**目录 + 开头若干块**，先看清这篇论文有什么；
    * 传 ``page``——返回与该页有交集的块（引用核对时用）；
    * 传 ``section``——按小节名做子串匹配（``§3 Method`` 这种定位串直接回喂）。

    三种都可以配 ``offset`` / ``limit`` 继续往下翻。返回里始终带 ``locator``，
    调用方拿到就能当出处写进回答，不需要自己拼页码。
    """
    from ..models.chunk import CHUNK_CODE

    paper = get_paper(paper_id)
    if paper is None:
        return {"error": "论文不存在"}

    # 正文块按 ord 排；代码块不属于「读论文正文」这条动线，排除掉
    query = db.session.query(Chunk).filter(
        Chunk.paper_id == paper_id,
        Chunk.note_id.is_(None),
        Chunk.kind != CHUNK_CODE,
    )
    if page is not None:
        # 与该页有交集：起始页 <= page 且结束页 >= page。
        # page_to 可能为空（老数据），只有 page_from 时按单页算。
        query = query.filter(
            Chunk.page_from <= page,
            or_(Chunk.page_to >= page, Chunk.page_to.is_(None), Chunk.page_from == page),
        )
    if section:
        query = query.filter(Chunk.section_path.ilike(f"%{section}%"))

    total = query.count()
    rows = (
        query.order_by(Chunk.ord)
        .offset(max(0, offset))
        .limit(max(1, min(limit, 50)))
        .all()
    )

    # 目录：只取小节起点，让调用方一眼看到全貌再决定读哪儿。
    # 单独查一次而不是从上面那页结果里推——翻到第 3 页时目录不该跟着变。
    outline = [
        {
            "section_path": path,
            "page_from": page_from,
            "ord": ord_,
        }
        for path, page_from, ord_ in db.session.query(
            Chunk.section_path, Chunk.page_from, Chunk.ord
        )
        .filter(
            Chunk.paper_id == paper_id,
            Chunk.note_id.is_(None),
            Chunk.kind != CHUNK_CODE,
            Chunk.is_section_start.is_(True),
        )
        .order_by(Chunk.ord)
        .all()
    ]

    chunks = []
    for row in rows:
        bits = []
        if row.section_path:
            bits.append(f"§{row.section_path}")
        if row.page_from:
            bits.append(
                f"p.{row.page_from}-{row.page_to}"
                if row.page_to and row.page_to != row.page_from
                else f"p.{row.page_from}"
            )
        chunks.append(
            {
                "chunk_id": row.id,
                "kind": row.kind,
                "section_path": row.section_path,
                "page_from": row.page_from,
                "page_to": row.page_to,
                "locator": " ".join(bits),
                "text": row.text,
            }
        )

    next_offset = offset + len(chunks)
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "outline": outline,
        "total_chunks": total,
        "offset": offset,
        "returned": len(chunks),
        "next_offset": next_offset if next_offset < total else None,
        "chunks": chunks,
    }


def stats() -> dict[str, int]:
    alive = db.session.query(Paper).filter(Paper.deleted_at.is_(None))
    return {
        "total": alive.count(),
        "unread": alive.filter(Paper.reading_status == "unread").count(),
        "reading": alive.filter(Paper.reading_status == "reading").count(),
        "read": alive.filter(Paper.reading_status == "read").count(),
        "pending": alive.filter(Paper.ingest_status == INGEST_PENDING).count(),
        "parsed": alive.filter(Paper.ingest_status == INGEST_PARSED).count(),
        "failed": alive.filter(Paper.ingest_status == INGEST_FAILED).count(),
        "deleted": db.session.query(Paper).filter(Paper.deleted_at.isnot(None)).count(),
    }


__all__ = [
    "PaperError",
    "create_from_path",
    "filter_options",
    "get_paper",
    "get_paper_by_path",
    "list_duplicates",
    "list_papers",
    "mark_reading",
    "paper_text",
    "purge",
    "resolve_duplicate",
    "soft_delete",
    "stats",
    "store_upload",
    "update_paper",
]
