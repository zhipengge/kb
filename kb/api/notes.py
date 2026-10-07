"""笔记接口。

对外提供的不只是 JSON——``Accept: text/markdown`` 时直接返回带 frontmatter
的 Markdown 原文。这对 agent 很重要：把笔记喂给模型时，Markdown 比
嵌套 JSON 省 token，也更接近模型训练时的格式。
"""

from __future__ import annotations

import logging

from flask import Response, request

from . import api_bp
from .auth import actor_name, require_scope
from .envelope import error_response, ok, paged, parse_paging

log = logging.getLogger(__name__)


@api_bp.get("/notes")
@require_scope("read")
def list_notes_endpoint():
    """笔记列表。支持按论文、类型、状态、标签、关键词筛选。"""
    from ..services import notes as notes_service

    limit, cursor = parse_paging()
    rows, next_cursor, total = notes_service.list_notes(
        query=request.args.get("q"),
        paper_id=request.args.get("paper_id"),
        kind=request.args.get("kind"),
        status=request.args.get("status"),
        tag_ids=request.args.getlist("tag") or None,
        standalone=request.args.get("standalone", type=lambda v: v.lower() == "true") or False,
        limit=limit,
        cursor=cursor,
        sort=request.args.get("sort", "updated"),
    )

    return paged(
        [notes_service.to_dict(note, include_content=False) for note in rows],
        next_cursor=next_cursor,
        limit=limit,
        total=total,
    )


@api_bp.get("/notes/<note_id>")
@require_scope("read")
def get_note_endpoint(note_id: str):
    from ..services import notes as notes_service

    note = notes_service.get_note(note_id)
    if note is None:
        return error_response("not_found", "笔记不存在", 404)

    # Accept: text/markdown 时返回 Markdown 原文
    if "text/markdown" in (request.headers.get("Accept") or ""):
        markdown = notes_service.export_markdown(note_id)
        return Response(markdown, mimetype="text/markdown; charset=utf-8")

    return ok(notes_service.to_dict(note))


@api_bp.post("/notes")
@require_scope("write")
def create_note_endpoint():
    """创建笔记。"""
    from ..services import notes as notes_service

    payload = request.get_json(silent=True) or {}
    try:
        note = notes_service.create_note(
            title=payload.get("title") or "",
            content_md=payload.get("content_md") or payload.get("content") or "",
            paper_id=payload.get("paper_id"),
            kind=payload.get("kind") or "manual",
            source="ai" if payload.get("source") == "ai" else "human",
            status=payload.get("status") or "draft",
            model=payload.get("model"),
            prompt_version=payload.get("prompt_version"),
        )
    except notes_service.NoteError as exc:
        return error_response("invalid_argument", str(exc), 400)

    _apply_tags(note, payload.get("tags") or [])
    return ok(notes_service.to_dict(note), status=201)


@api_bp.patch("/notes/<note_id>")
@api_bp.put("/notes/<note_id>")
@require_scope("write")
def update_note_endpoint(note_id: str):
    """更新笔记。

    支持乐观锁：带上 ``If-Match: <版本号>`` 或请求体里的 ``version``。
    版本不一致会返回 409 而不是覆盖——并发编辑时静默丢内容是不可接受的。
    """
    from ..services import notes as notes_service

    payload = request.get_json(silent=True) or {}

    expected = None
    if_match = request.headers.get("If-Match")
    if if_match:
        try:
            expected = int(if_match.strip('"'))
        except ValueError:
            return error_response("invalid_argument", "If-Match 需要是一个版本号", 400)
    elif payload.get("version") is not None:
        expected = int(payload["version"])

    values = {
        key: payload[key]
        for key in ("title", "content_md", "kind", "status")
        if key in payload
    }

    try:
        note = notes_service.update_note(
            note_id, values, author=actor_name(), expected_version=expected
        )
    except notes_service.NoteError as exc:
        message = str(exc)
        code = "conflict" if "已被其他操作修改" in message else "invalid_argument"
        status = 409 if code == "conflict" else 400
        return error_response(code, message, status)

    _apply_tags(note, payload.get("tags") or [])
    return ok(notes_service.to_dict(note))


@api_bp.delete("/notes/<note_id>")
@require_scope("write")
def delete_note_endpoint(note_id: str):
    """删除笔记。

    默认**保留磁盘文件**——用户可能还在 Obsidian 里用它。
    ``?remove_file=true`` 才连文件一起删。
    """
    from ..services import notes as notes_service

    try:
        notes_service.delete_note(
            note_id, remove_file=request.args.get("remove_file", type=lambda v: v.lower() == "true")
        )
    except notes_service.NoteError as exc:
        return error_response("not_found", str(exc), 404)
    return ok({"id": note_id, "deleted": True})


@api_bp.get("/notes/<note_id>/revisions")
@require_scope("read")
def list_revisions_endpoint(note_id: str):
    """版本历史。"""
    from ..services import notes as notes_service
    from ..utils.time import iso

    note = notes_service.get_note(note_id)
    if note is None:
        return error_response("not_found", "笔记不存在", 404)

    revisions = notes_service.list_revisions(note_id)
    return ok(
        {
            "current_version": note.version,
            "revisions": [
                {
                    "version": rev.version,
                    "title": rev.title,
                    "author": rev.author,
                    "summary": rev.summary,
                    "created_at": iso(rev.created_at),
                }
                for rev in revisions
            ],
        }
    )


@api_bp.get("/notes/<note_id>/diff")
@require_scope("read")
def diff_note_endpoint(note_id: str):
    """版本对比。``from`` 与 ``to`` 省略 to 时对比到当前内容。"""
    from ..services import notes as notes_service

    from_version = request.args.get("from", type=int)
    to_version = request.args.get("to", type=int)
    if from_version is None:
        return error_response("invalid_argument", "缺少 from 参数", 400)

    try:
        text = notes_service.diff_revisions(note_id, from_version, to_version)
    except notes_service.NoteError as exc:
        return error_response("not_found", str(exc), 404)

    return ok({"diff": text})


@api_bp.post("/notes/<note_id>/restore/<int:version>")
@require_scope("write")
def restore_note_endpoint(note_id: str, version: int):
    """回滚到历史版本。"""
    from ..services import notes as notes_service

    try:
        note = notes_service.restore_revision(note_id, version)
    except notes_service.NoteError as exc:
        return error_response("invalid_argument", str(exc), 400)
    return ok(notes_service.to_dict(note))


@api_bp.post("/notes/sync")
@require_scope("write")
def sync_notes_endpoint():
    """触发与磁盘的同步。

    ``pull_only=true`` 时只把外部改动读回来，不往磁盘写任何东西。
    """
    from ..services import notesync

    payload = request.get_json(silent=True) or {}
    pull_only = bool(
        request.args.get("pull_only", type=lambda v: v.lower() == "true")
        or payload.get("pull_only")
    )

    result = notesync.sync_all(pull_only=pull_only)
    return ok(result)


@api_bp.get("/notes/sync/status")
@require_scope("read")
def sync_status_endpoint():
    """同步状态概览：哪些笔记有冲突、缺失、待回写。"""
    from ..extensions import db
    from ..models import Note

    rows = (
        db.session.query(Note.sync_state, db.func.count(Note.id))
        .group_by(Note.sync_state)
        .all()
    )
    counts = {state or "unknown": count for state, count in rows}
    total = db.session.query(Note).count()

    return ok(
        {
            "total": total,
            "by_state": counts,
            "needs_attention": sum(
                counts.get(state, 0) for state in ("conflict", "missing", "error")
            ),
        }
    )


@api_bp.get("/notes/untracked")
@require_scope("read")
def untracked_notes_endpoint():
    """笔记目录里存在、但数据库里还没有记录的 Markdown 文件。

    这是「用户在 Obsidian 里直接新建笔记」的入口。列出来而不是自动导入：
    自动导入会产生一批来源不明的记录，让用户先看一眼更稳妥。
    """
    from ..services import notesync

    return ok(notesync.find_untracked_files())


def _apply_tags(note, names: list) -> None:
    """把请求里带的标签附加到笔记上。"""
    if not names:
        return
    from ..services import tagging

    for name in names:
        if isinstance(name, str) and name.strip():
            try:
                tagging.attach_tag(note, name.strip(), source="manual")
            except Exception:
                log.warning("附加标签 %r 失败", name, exc_info=True)
