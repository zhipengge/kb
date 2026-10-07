"""标签接口：词表读写与 AI 建议裁决。

对 agent 来说最重要的是 ``/tags/suggest`` 与 ``/tags/suggestions/{id}/resolve``
这一对：它们让外部 agent 可以像内置的深度阅读流水线一样提出标签建议，
但仍然把最终决定权留给人。这个「提议权与决定权分离」的设计是词表不失控的前提。
"""

from __future__ import annotations

from flask import request

from . import api_bp
from .auth import require_scope
from .envelope import error_response, ok


def _tag_dict(tag) -> dict:
    return {
        "id": tag.id,
        "name": tag.name,
        "slug": tag.slug,
        "dimension": tag.dimension,
        "color": tag.color,
        "description": tag.description,
        "aliases": tag.aliases or [],
        "usage_count": tag.usage_count,
        "parent_id": tag.parent_id,
        "is_auto": tag.is_auto,
    }


@api_bp.get("/tags")
@require_scope("read")
def list_tags_endpoint():
    """列出标签词表。

    ``tree=true`` 时按维度返回层级结构，否则返回扁平列表。
    """
    from ..services import tagging

    if request.args.get("tree", type=lambda v: v.lower() == "true"):
        return ok(tagging.tag_tree())

    tags = tagging.list_tags(
        dimension=request.args.get("dimension"),
        query=request.args.get("q"),
    )
    return ok([_tag_dict(tag) for tag in tags])


@api_bp.post("/tags")
@require_scope("write")
def create_tag_endpoint():
    """创建标签。已存在同名（含别名）时返回既有的那个，不报错。

    这是刻意的：调用方的意图是「确保这个标签存在」，
    因重复创建而报错会迫使每个调用点都先查一次。
    """
    from ..services import tagging

    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return error_response("invalid_argument", "缺少标签名 name", 400)

    try:
        tag = tagging.get_or_create_tag(
            name,
            dimension=payload.get("dimension") or "misc",
            source="manual",
        )
    except tagging.TagError as exc:
        return error_response("invalid_argument", str(exc), 400)

    if payload.get("color"):
        tag.color = payload["color"]
    if payload.get("description"):
        tag.description = payload["description"]
    if payload.get("aliases"):
        from ..extensions import db

        aliases = [a for a in payload["aliases"] if isinstance(a, str) and a.strip()]
        if aliases:
            tag.aliases = sorted(set((tag.aliases or []) + aliases))
    from ..extensions import db

    db.session.commit()

    return ok(_tag_dict(tag), status=201)


@api_bp.patch("/tags/<tag_id>")
@require_scope("write")
def update_tag_endpoint(tag_id: str):
    """修改标签的名称、别名、颜色、父节点。"""
    from ..extensions import db
    from ..models import Tag

    tag = db.session.get(Tag, tag_id)
    if tag is None:
        return error_response("not_found", "标签不存在", 404)

    payload = request.get_json(silent=True) or {}

    if "name" in payload and str(payload["name"]).strip():
        tag.name = str(payload["name"]).strip()[:255]
    if "color" in payload:
        tag.color = payload["color"]
    if "description" in payload:
        tag.description = payload["description"]
    if "parent_id" in payload:
        parent_id = payload["parent_id"]
        if parent_id == tag.id:
            return error_response("invalid_argument", "标签不能以自己为父节点", 400)
        if parent_id and db.session.get(Tag, parent_id) is None:
            return error_response("invalid_argument", "父标签不存在", 400)
        tag.parent_id = parent_id
    if "aliases" in payload:
        aliases = [a for a in payload["aliases"] if isinstance(a, str) and a.strip()]
        tag.aliases = sorted(set(aliases))

    db.session.commit()
    return ok(_tag_dict(tag))


@api_bp.delete("/tags/<tag_id>")
@require_scope("write")
def delete_tag_endpoint(tag_id: str):
    """删除标签。所有关联一并解除。"""
    from ..extensions import db
    from ..models import Tag

    tag = db.session.get(Tag, tag_id)
    if tag is None:
        return error_response("not_found", "标签不存在", 404)

    name = tag.name
    db.session.delete(tag)
    db.session.commit()
    return ok({"id": tag_id, "name": name, "deleted": True})


@api_bp.get("/tags/<tag_id>/papers")
@require_scope("read")
def tag_papers_endpoint(tag_id: str):
    """某标签下的论文。"""
    from ..extensions import db
    from ..models import Paper, PaperTag
    from .papers import paper_dict

    rows = (
        db.session.query(Paper)
        .join(PaperTag, PaperTag.paper_id == Paper.id)
        .filter(PaperTag.tag_id == tag_id, Paper.deleted_at.is_(None))
        .order_by(Paper.id.desc())
        .limit(200)
        .all()
    )
    return ok([paper_dict(paper) for paper in rows])


# --------------------------------------------------------------------------
# 实体打标
# --------------------------------------------------------------------------


@api_bp.post("/papers/<paper_id>/tags")
@require_scope("write")
def tag_paper_endpoint(paper_id: str):
    """给论文打标签 / 取消标签。

    请求体：``{"add": ["方法名"], "remove": ["tag_id"], "dimension": "method"}``
    """
    from ..extensions import db
    from ..models import Paper
    from ..services import tagging

    paper = db.session.get(Paper, paper_id)
    if paper is None:
        return error_response("not_found", "论文不存在", 404)

    payload = request.get_json(silent=True) or {}
    dimension = payload.get("dimension") or "misc"

    added = []
    for name in payload.get("add") or []:
        if not isinstance(name, str) or not name.strip():
            continue
        tag = tagging.attach_tag(paper, name.strip(), dimension=dimension, source="manual")
        if tag is not None:
            added.append(_tag_dict(tag))

    removed = 0
    for tag_id in payload.get("remove") or []:
        if tagging.detach_tag(paper, tag_id):
            removed += 1

    db.session.refresh(paper)
    return ok(
        {
            "paper_id": paper_id,
            "added": added,
            "removed": removed,
            "tags": [_tag_dict(link.tag) for link in paper.tag_links if link.tag],
        }
    )


@api_bp.post("/notes/<note_id>/tags")
@require_scope("write")
def tag_note_endpoint(note_id: str):
    """给笔记打标签 / 取消标签。"""
    from ..extensions import db
    from ..models import Note
    from ..services import tagging

    note = db.session.get(Note, note_id)
    if note is None:
        return error_response("not_found", "笔记不存在", 404)

    payload = request.get_json(silent=True) or {}
    dimension = payload.get("dimension") or "misc"

    added = []
    for name in payload.get("add") or []:
        if not isinstance(name, str) or not name.strip():
            continue
        tag = tagging.attach_tag(note, name.strip(), dimension=dimension, source="manual")
        if tag is not None:
            added.append(_tag_dict(tag))

    removed = 0
    for tag_id in payload.get("remove") or []:
        if tagging.detach_tag(note, tag_id):
            removed += 1

    db.session.refresh(note)
    return ok(
        {
            "note_id": note_id,
            "added": added,
            "removed": removed,
            "tags": [_tag_dict(link.tag) for link in note.tag_links if link.tag],
        }
    )


# --------------------------------------------------------------------------
# AI 建议队列
# --------------------------------------------------------------------------


@api_bp.get("/tags/suggestions")
@require_scope("read")
def list_suggestions_endpoint():
    """待裁决的标签建议。"""
    from ..services import tagging
    from ..utils.time import iso

    status = request.args.get("status", "pending")
    rows = tagging.list_suggestions(status)
    return ok(
        {
            "stats": tagging.suggestion_stats(),
            "items": [
                {
                    "id": row.id,
                    "name": row.name,
                    "dimension": row.dimension,
                    "paper_id": row.paper_id,
                    "note_id": row.note_id,
                    "rationale": row.rationale,
                    "confidence": row.confidence,
                    "model": row.model,
                    "suggested_tag_id": row.suggested_tag_id,
                    "status": row.status,
                    "created_at": iso(row.created_at),
                }
                for row in rows
            ],
        }
    )


@api_bp.post("/tags/suggest")
@require_scope("write")
def suggest_tag_endpoint():
    """提交一条标签建议（供外部 agent 使用）。

    注意这**不会**让标签进入词表——只进建议队列。裁定权始终在人手里，
    否则多个 agent 并发写标签会迅速把词表搞乱。
    """
    from ..services import tagging

    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return error_response("invalid_argument", "缺少标签名 name", 400)
    if not payload.get("paper_id") and not payload.get("note_id"):
        return error_response("invalid_argument", "必须提供 paper_id 或 note_id", 400)

    try:
        suggestion = tagging.suggest_tag(
            name=name,
            dimension=payload.get("dimension") or "misc",
            paper_id=payload.get("paper_id"),
            note_id=payload.get("note_id"),
            rationale=payload.get("rationale"),
            confidence=payload.get("confidence"),
            model=payload.get("model"),
        )
    except tagging.TagError as exc:
        return error_response("invalid_argument", str(exc), 400)

    if suggestion is None:
        # 已被拒绝过、或已有同样的待裁决建议——都不是错误
        return ok({"created": False, "reason": "已存在待裁决建议，或该建议此前被拒绝过"})

    return ok({"created": True, "id": suggestion.id, "name": suggestion.name}, status=201)


@api_bp.post("/tags/suggestions/<suggestion_id>/resolve")
@require_scope("write")
def resolve_suggestion_endpoint(suggestion_id: str):
    """裁决标签建议。``{"action": "accept" | "reject"}``"""
    from ..services import tagging

    payload = request.get_json(silent=True) or {}
    action = payload.get("action")
    if action not in {"accept", "reject"}:
        return error_response("invalid_argument", "action 必须是 accept 或 reject", 400)

    try:
        tag = tagging.resolve_suggestion(suggestion_id, action, dimension=payload.get("dimension"))
    except tagging.TagError as exc:
        return error_response("invalid_argument", str(exc), 400)

    return ok({"id": suggestion_id, "action": action, "tag": _tag_dict(tag) if tag else None})


@api_bp.post("/tags/suggestions/accept-all")
@require_scope("write")
def accept_all_suggestions_endpoint():
    """批量接受建议，可按 ``min_confidence`` 过滤。"""
    from ..services import tagging

    payload = request.get_json(silent=True) or {}
    accepted = tagging.accept_all_suggestions(
        min_confidence=float(payload.get("min_confidence") or 0.0)
    )
    return ok({"accepted": accepted, "stats": tagging.suggestion_stats()})
