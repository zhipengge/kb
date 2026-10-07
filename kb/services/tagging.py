"""标签体系：受控词表、关联、AI 建议队列。

这个模块存在的理由是**防止标签爆炸**。如果让模型自由打标，几百篇论文之后
你会得到几千个只用过一次的标签（"attention"、"Attention"、"注意力"、
"attention mechanism"…），标签就从「聚合工具」退化成「噪音」。

三道闸门：

  1. **归一到已有标签。** 附加标签时先按名称、别名、slug 匹配现有词表；
     匹配上了就不再新建。模型写 "self-attention"、用户写 "自注意力"，
     只要它们被登记为别名，就都指向同一个标签。
  2. **AI 只能建议，不能直接创建。** 模型的提议进 ``tag_suggestions``
     等人裁决。拒绝也记录下来，同一个词下次不再打扰。
  3. **维度隔离。** "Transformer" 作为*方法*和作为*架构*是两回事，
     同一个词在不同维度下是两个标签。这避免了不同语义的混用。
"""

from __future__ import annotations

import logging

from sqlalchemy import func, or_

from ..extensions import db
from ..models import Note, Paper, Tag, TagSuggestion
from ..models.base import utcnow
from ..models.tag import (
    DIM_MISC,
    DIMENSIONS,
    SOURCE_AI,
    SOURCE_MANUAL,
    NoteTag,
    PaperTag,
)
from .paths import slugify

log = logging.getLogger(__name__)


class TagError(ValueError):
    """标签操作失败。消息面向用户。"""


# --------------------------------------------------------------------------
# 词表查询
# --------------------------------------------------------------------------


def find_tag(name: str, dimension: str | None = None) -> Tag | None:
    """按名称 / 别名 / slug 查找标签。

    匹配顺序体现优先级：精确名称 > 别名 > slug。名称最不容易误伤，
    slug 是归一化后的字符串（"attention-mechanism"），匹配面最宽。
    """
    if not name or not name.strip():
        return None

    cleaned = name.strip()
    slug = slugify(cleaned, fallback="")

    query = db.session.query(Tag)
    if dimension:
        query = query.filter(Tag.dimension == dimension)

    exact = query.filter(func.lower(Tag.name) == cleaned.lower()).first()
    if exact is not None:
        return exact

    if slug:
        by_slug = query.filter(Tag.slug == slug).first()
        if by_slug is not None:
            return by_slug

    # 别名是 JSON 数组，SQLite 上没法直接索引查询，只能取出来比。
    # 标签总数是几百到几千量级，全表扫一次完全可以接受。
    lowered = cleaned.lower()
    for tag in query.all():
        for alias in tag.aliases or []:
            if isinstance(alias, str) and alias.strip().lower() == lowered:
                return tag

    return None


def get_or_create_tag(
    name: str,
    *,
    dimension: str = DIM_MISC,
    source: str = SOURCE_MANUAL,
    add_alias: bool = False,
) -> Tag:
    """取得标签，不存在则创建。

    ``add_alias=True`` 时，如果名字匹配到了已有标签但不是它的正式名称，
    就把这个名字登记为别名——下次再遇到同样的写法就能直接命中，
    不必再走一遍全表比对。
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise TagError("标签名不能为空")
    if len(cleaned) > 120:
        raise TagError("标签名过长（最多 120 字符）")
    if dimension not in DIMENSIONS:
        dimension = DIM_MISC

    existing = find_tag(cleaned, dimension)
    if existing is not None:
        if add_alias and existing.name.lower() != cleaned.lower():
            aliases = list(existing.aliases or [])
            if cleaned not in aliases:
                aliases.append(cleaned)
                existing.aliases = aliases
                db.session.commit()
        return existing

    slug = slugify(cleaned, fallback="tag")
    # slug 在同一维度下必须唯一，重名时加后缀
    base_slug, suffix = slug, 2
    while (
        db.session.query(Tag.id)
        .filter(Tag.dimension == dimension, Tag.slug == slug)
        .first()
        is not None
    ):
        slug = f"{base_slug}-{suffix}"
        suffix += 1

    tag = Tag(
        name=cleaned,
        slug=slug,
        dimension=dimension,
        aliases=[],
        is_auto=(source == SOURCE_AI),
    )
    db.session.add(tag)
    db.session.commit()
    log.info("已创建标签 %s:%s", dimension, cleaned)
    return tag


def list_tags(
    *, dimension: str | None = None, query: str | None = None, with_counts: bool = False
) -> list[Tag]:
    statement = db.session.query(Tag)
    if dimension:
        statement = statement.filter(Tag.dimension == dimension)
    if query:
        pattern = f"%{query.strip()}%"
        statement = statement.filter(or_(Tag.name.ilike(pattern), Tag.slug.ilike(pattern)))
    return statement.order_by(Tag.dimension, Tag.usage_count.desc(), Tag.name).all()


def tag_tree() -> dict[str, list[dict]]:
    """按维度分组的层级标签树，供筛选器与设置页渲染。

    返回 ``{维度: [根节点, ...]}``。每个维度只包含属于它的标签——
    按维度隔离是这个词表能被有效筛选的前提。
    """
    tags = db.session.query(Tag).order_by(Tag.dimension, Tag.name).all()

    # 按 (维度, 父节点) 建索引。用维度做第一层键，避免不同维度下
    # 恰好同名的子树串到一起。
    children: dict[tuple[str, str | None], list[Tag]] = {}
    for tag in tags:
        children.setdefault((tag.dimension, tag.parent_id), []).append(tag)

    def build(dimension: str, parent_id: str | None, depth: int = 0) -> list[dict]:
        nodes = []
        for tag in children.get((dimension, parent_id), []):
            nodes.append(
                {
                    "id": tag.id,
                    "name": tag.name,
                    "slug": tag.slug,
                    "dimension": tag.dimension,
                    "color": tag.color,
                    "usage_count": tag.usage_count,
                    "depth": depth,
                    "children": build(dimension, tag.id, depth + 1),
                }
            )
        return nodes

    return {dimension: build(dimension, None) for dimension in DIMENSIONS}


# --------------------------------------------------------------------------
# 关联
# --------------------------------------------------------------------------


def _recount(tag_id: str) -> None:
    """重算使用计数。

    计数字段是冗余的（标签云要按热度排序，每次 COUNT 太慢），
    所以在增删关联时必须维护。这里用重算而不是自增，因为自增在
    并发与异常路径下容易漂移，而重算永远是对的。
    """
    count = (
        db.session.query(func.count(PaperTag.tag_id)).filter(PaperTag.tag_id == tag_id).scalar()
        or 0
    ) + (
        db.session.query(func.count(NoteTag.tag_id)).filter(NoteTag.tag_id == tag_id).scalar()
        or 0
    )
    db.session.query(Tag).filter(Tag.id == tag_id).update(
        {"usage_count": count}, synchronize_session=False
    )


def attach_tag(
    target: Paper | Note,
    name: str,
    *,
    dimension: str = DIM_MISC,
    source: str = SOURCE_MANUAL,
    confidence: float | None = None,
) -> Tag | None:
    """给论文或笔记打标签。

    会先尝试匹配现有标签（含别名），匹配不到才新建。这是「受控」的关键：
    调用方可以放心地把模型输出的任意字符串传进来，归一化在这里完成。
    """
    if not name or not str(name).strip():
        return None

    tag = get_or_create_tag(name, dimension=dimension, source=source, add_alias=True)

    if isinstance(target, Paper):
        exists = (
            db.session.query(PaperTag)
            .filter_by(paper_id=target.id, tag_id=tag.id)
            .one_or_none()
        )
        if exists is None:
            db.session.add(
                PaperTag(
                    paper_id=target.id, tag_id=tag.id, source=source, confidence=confidence
                )
            )
    else:
        exists = (
            db.session.query(NoteTag)
            .filter_by(note_id=target.id, tag_id=tag.id)
            .one_or_none()
        )
        if exists is None:
            db.session.add(
                NoteTag(note_id=target.id, tag_id=tag.id, source=source, confidence=confidence)
            )

    db.session.commit()
    _recount(tag.id)
    db.session.commit()
    return tag


def detach_tag(target: Paper | Note, tag_id: str) -> bool:
    if isinstance(target, Paper):
        removed = (
            db.session.query(PaperTag)
            .filter_by(paper_id=target.id, tag_id=tag_id)
            .delete(synchronize_session=False)
        )
    else:
        removed = (
            db.session.query(NoteTag)
            .filter_by(note_id=target.id, tag_id=tag_id)
            .delete(synchronize_session=False)
        )
    db.session.commit()
    if removed:
        _recount(tag_id)
        db.session.commit()
    return bool(removed)


def recompute_all_counts() -> int:
    """全量重算使用计数。用于修复漂移或迁移后。"""
    tags = db.session.query(Tag).all()
    for tag in tags:
        _recount(tag.id)
    db.session.commit()
    return len(tags)


# --------------------------------------------------------------------------
# AI 建议队列
# --------------------------------------------------------------------------


def suggest_tag(
    *,
    name: str,
    dimension: str = DIM_MISC,
    paper_id: str | None = None,
    note_id: str | None = None,
    rationale: str | None = None,
    confidence: float | None = None,
    model: str | None = None,
    evidence: dict | None = None,
) -> TagSuggestion | None:
    """记录一条 AI 建议。

    两个去重点：已存在的待裁决建议不重复插入；**被拒绝过的词不再提**——
    否则模型每次跑都提同一个词，用户要反复拒绝同一件事。
    """
    cleaned = (name or "").strip()
    if not cleaned or len(cleaned) > 120:
        return None
    if not paper_id and not note_id:
        raise TagError("建议必须关联到论文或笔记")

    rejected = (
        db.session.query(TagSuggestion.id)
        .filter(
            TagSuggestion.name == cleaned,
            TagSuggestion.status == "rejected",
        )
        .first()
    )
    if rejected is not None:
        return None

    pending = (
        db.session.query(TagSuggestion.id)
        .filter(
            TagSuggestion.name == cleaned,
            TagSuggestion.status == "pending",
            TagSuggestion.paper_id == paper_id,
            TagSuggestion.note_id == note_id,
        )
        .first()
    )
    if pending is not None:
        return None

    # 如果这个词其实已经是某个标签的别名，记下来——界面上可以显示成
    # 「建议：自注意力（已有标签「注意力机制」的别名）」，而不是让人以为要新建
    matched = find_tag(cleaned, dimension)

    suggestion = TagSuggestion(
        paper_id=paper_id,
        note_id=note_id,
        name=cleaned,
        dimension=dimension if dimension in DIMENSIONS else DIM_MISC,
        suggested_tag_id=matched.id if matched else None,
        rationale=(rationale or "")[:1000] or None,
        confidence=confidence,
        model=model,
        evidence=evidence or {},
        status="pending",
    )
    db.session.add(suggestion)
    db.session.commit()
    return suggestion


def list_suggestions(status: str = "pending", limit: int = 200) -> list[TagSuggestion]:
    return (
        db.session.query(TagSuggestion)
        .filter(TagSuggestion.status == status)
        .order_by(TagSuggestion.confidence.desc().nullslast(), TagSuggestion.created_at.desc())
        .limit(limit)
        .all()
    )


def resolve_suggestion(suggestion_id: str, action: str, *, dimension: str | None = None) -> Tag | None:
    """裁决一条建议。

    ``accept``  把标签真正关联到目标上（复用 attach_tag，所以同样会归一到已有标签）
    ``reject``  标记拒绝，同一个词以后不再提
    """
    suggestion = db.session.get(TagSuggestion, suggestion_id)
    if suggestion is None:
        raise TagError("建议不存在")
    if suggestion.status != "pending":
        raise TagError(f"这条建议已经处理过了（{suggestion.status}）")

    if action == "reject":
        suggestion.status = "rejected"
        suggestion.resolved_at = utcnow()
        db.session.commit()
        return None

    if action != "accept":
        raise TagError(f"未知的处理方式：{action}")

    target: Paper | Note | None = None
    if suggestion.paper_id:
        target = db.session.get(Paper, suggestion.paper_id)
    elif suggestion.note_id:
        target = db.session.get(Note, suggestion.note_id)
    if target is None:
        raise TagError("建议关联的对象已不存在")

    tag = attach_tag(
        target,
        suggestion.name,
        dimension=dimension or suggestion.dimension,
        source=SOURCE_AI,
        confidence=suggestion.confidence,
    )

    suggestion.status = "accepted"
    suggestion.resolved_at = utcnow()
    db.session.commit()
    return tag


def accept_all_suggestions(*, min_confidence: float = 0.0) -> int:
    """批量接受建议（可按置信度阈值过滤）。"""
    pending = list_suggestions("pending")
    accepted = 0
    for suggestion in pending:
        if (suggestion.confidence or 1.0) < min_confidence:
            continue
        try:
            resolve_suggestion(suggestion.id, "accept")
            accepted += 1
        except TagError:
            continue
    log.info("批量接受了 %d 条标签建议", accepted)
    return accepted


def suggestion_stats() -> dict:
    rows = (
        db.session.query(TagSuggestion.status, func.count(TagSuggestion.id))
        .group_by(TagSuggestion.status)
        .all()
    )
    counts = {status: count for status, count in rows}
    return {
        "pending": counts.get("pending", 0),
        "accepted": counts.get("accepted", 0),
        "rejected": counts.get("rejected", 0),
    }


def seed_dimension_defaults() -> None:
    """给未设置维度的标签补一个合理默认。

    早期数据或手工 SQL 插入的标签可能没有维度。没有维度意味着
    筛选器里看不到它们——用户会以为标签丢了。
    """
    updated = (
        db.session.query(Tag)
        .filter(or_(Tag.dimension.is_(None), Tag.dimension == ""))
        .update({"dimension": DIM_MISC}, synchronize_session=False)
    )
    if updated:
        db.session.commit()
        log.info("已为 %d 个标签补上默认维度", updated)


__all__ = [
    "TagError",
    "accept_all_suggestions",
    "attach_tag",
    "detach_tag",
    "find_tag",
    "get_or_create_tag",
    "list_suggestions",
    "list_tags",
    "recompute_all_counts",
    "resolve_suggestion",
    "seed_dimension_defaults",
    "suggest_tag",
    "suggestion_stats",
    "tag_tree",
]
