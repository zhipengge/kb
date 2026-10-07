"""笔记的增删改查。

笔记是本系统里最贵的数据：论文能重新下载，索引能重建，人写的理解丢了就没了。
所以这里的每个设计都偏向「宁可多留一份，不要少留一份」：

  * 任何内容变更前先存版本快照（``note_revisions``）；
  * 删除默认软删除；
  * 与磁盘冲突时不自动解决；
  * 写入磁盘失败不影响数据库里的内容——数据库才是权威副本。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from sqlalchemy import or_

from ..extensions import db
from ..models import Note, NoteRevision, Paper
from ..models.note import (
    KIND_MANUAL,
    KINDS,
    SOURCE_HUMAN,
    STATUS_DRAFT,
    STATUSES,
)
from .paths import slugify

log = logging.getLogger(__name__)


class NoteError(ValueError):
    """笔记操作失败。消息面向用户。"""


# --------------------------------------------------------------------------
# 查询
# --------------------------------------------------------------------------

_SORTS = {
    "updated": Note.updated_at.desc(),
    "created": Note.created_at.desc(),
    "title": Note.title.asc(),
}


def list_notes(
    *,
    query: str | None = None,
    paper_id: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    tag_ids: list[str] | None = None,
    standalone: bool = False,
    limit: int = 50,
    cursor: str | None = None,
    sort: str = "updated",
) -> tuple[list[Note], str | None, int]:
    """查询笔记。返回 ``(结果, 下一页游标, 总数)``。"""
    from ..models import NoteTag

    base = db.session.query(Note)

    if paper_id:
        base = base.filter(Note.paper_id == paper_id)
    elif standalone:
        # 「独立笔记」——不属于任何论文的读书笔记、技术总结
        base = base.filter(Note.paper_id.is_(None))

    if kind and kind in KINDS:
        base = base.filter(Note.kind == kind)
    if status and status in STATUSES:
        base = base.filter(Note.status == status)

    if query:
        pattern = f"%{query.strip()}%"
        base = base.filter(
            or_(Note.title.ilike(pattern), Note.content_md.ilike(pattern))
        )

    if tag_ids:
        for tag_id in tag_ids:
            base = base.filter(
                Note.id.in_(db.session.query(NoteTag.note_id).filter(NoteTag.tag_id == tag_id))
            )

    total = base.count()
    ordered = base.order_by(_SORTS.get(sort, _SORTS["updated"]))
    if cursor:
        ordered = ordered.filter(Note.id < cursor)

    rows = ordered.limit(limit + 1).all()
    next_cursor = rows[limit].id if len(rows) > limit else None
    return rows[:limit], next_cursor, total


def get_note(note_id: str) -> Note | None:
    return db.session.get(Note, note_id)


def get_or_404(note_id: str) -> Note:
    note = get_note(note_id)
    if note is None:
        raise NoteError("笔记不存在")
    return note


def find_by_kb_id(kb_id: str) -> Note | None:
    return db.session.query(Note).filter(Note.id == kb_id).one_or_none()


# --------------------------------------------------------------------------
# 修改
# --------------------------------------------------------------------------

# prompt_version / model 也是可更新的：AI 重跑同一篇论文时是**覆盖**已有笔记，
# 如果这两个字段不跟着更新，笔记内容是新版、溯源信息还停在旧版号，
# 「这条内容是谁生成的」就查不出来了。
_EDITABLE = {"title", "content_md", "kind", "status", "prompt_version", "model"}


def _reindex(note: Note, changed: Iterable[str] | None = None) -> None:
    """笔记内容变了就重建它的检索块。

    只判断「要不要重建」：改个状态、加个标签不该触发一次分块。
    真正干活的是 ``indexer.reindex_note_safely``，那边的契约是**失败不抛**。

    ``changed`` 接受任何可迭代对象。调用方传的是 list（``update_note`` 里
    累积出来的），这里必须自己转成 set 再比——直接写 ``changed & {...}``
    会在 list 上抛 TypeError，而那个异常会把**保存**打断。
    """
    try:
        if changed is not None and not (set(changed) & {"content_md", "title"}):
            return
        from .indexer import reindex_note_safely

        reindex_note_safely(note)
    except Exception:
        # 索引是「能不能搜到」的辅助，保存是主操作。这里再兜一层：
        # 上面那句判断本身就曾经抛过一次 TypeError，把保存打回了 500。
        log.warning("笔记 %s 重建索引失败（内容已保存）", note.id, exc_info=True)


def _snapshot(note: Note, *, author: str, summary: str) -> None:
    """保存当前内容的快照。

    在**变更之前**调用。版本号用当前的 note.version，改动完成后
    note.version 自增——这样每个版本号对应一份确定的快照。
    """
    exists = (
        db.session.query(NoteRevision)
        .filter_by(note_id=note.id, version=note.version)
        .one_or_none()
    )
    if exists is not None:
        return
    db.session.add(
        NoteRevision(
            note_id=note.id,
            version=note.version,
            title=note.title,
            content_md=note.content_md,
            author=author,
            summary=summary[:500],
        )
    )


def update_note(
    note_id: str,
    values: dict,
    *,
    author: str = "human",
    write_disk: bool = True,
    expected_version: int | None = None,
) -> Note:
    """更新笔记。

    ``expected_version`` 用于乐观锁：网页端提交时带上它读到的版本号，
    如果不匹配说明期间有别的改动（另一个标签页、AI 任务、外部同步），
    这时应该拒绝而不是盲目覆盖。
    """
    note = get_or_404(note_id)

    if expected_version is not None and note.version != expected_version:
        raise NoteError(
            f"笔记已被其他操作修改（当前版本 {note.version}，你提交的是 {expected_version}）。"
            "请刷新后重新编辑。"
        )

    # 先算出要改什么，**不要边比较边赋值**——快照必须在内容被覆盖之前拍。
    # 早期版本在这里先赋值再快照，结果每一版快照存的都是「改完之后」的内容，
    # 版本历史退化成了一串重复的副本，回滚拿不回任何东西。
    changed: list[str] = []
    pending: dict[str, Any] = {}
    for key, value in values.items():
        if key not in _EDITABLE:
            continue
        if key == "kind" and value not in KINDS:
            raise NoteError(f"无效的笔记类型：{value}")
        if key == "status" and value not in STATUSES:
            raise NoteError(f"无效的状态：{value}")
        if getattr(note, key) != value:
            changed.append(key)
            pending[key] = value

    if not changed:
        return note

    _snapshot(note, author=author, summary=f"修改了 {', '.join(changed)}")

    for key, value in pending.items():
        setattr(note, key, value)

    # slug 跟着标题走，但只在用户没手工改过的情况下——
    # 手工改过的 slug 意味着文件名是被刻意定的，跟着标题变会打乱用户预期
    if "title" in changed and not note.slug:
        note.slug = slugify(note.title, fallback=note.id[:12])

    note.version += 1
    db.session.commit()

    if write_disk:
        _try_write(note)
    _reindex(note, changed)
    return note


def create_note(
    *,
    title: str = "",
    content_md: str = "",
    paper_id: str | None = None,
    kind: str = KIND_MANUAL,
    source: str = SOURCE_HUMAN,
    status: str = STATUS_DRAFT,
    slug: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    write_disk: bool = True,
    meta: dict | None = None,
) -> Note:
    """创建笔记。

    slug 在同一篇论文下必须唯一——它是落盘文件名的一部分，
    重名会让两篇笔记抢同一个文件。
    """
    if paper_id and db.session.get(Paper, paper_id) is None:
        raise NoteError("关联的论文不存在")

    base_slug = slugify(slug or title or "note", fallback="note")
    unique_slug = _unique_slug(paper_id, base_slug)

    note = Note(
        paper_id=paper_id,
        title=title or "未命名笔记",
        slug=unique_slug,
        content_md=content_md,
        kind=kind if kind in KINDS else KIND_MANUAL,
        status=status if status in STATUSES else STATUS_DRAFT,
        source=source,
        model=model,
        prompt_version=prompt_version,
        meta=meta or {},
    )
    db.session.add(note)
    db.session.commit()

    if write_disk:
        _try_write(note)
    _reindex(note, {"content_md"})

    log.info("已创建笔记 %s（%s）", note.id, note.title[:40])
    return note


def _unique_slug(paper_id: str | None, base: str) -> str:
    """确保 slug 在论文范围内唯一，必要时加数字后缀。"""
    candidate = base
    suffix = 2
    while True:
        query = db.session.query(Note.id).filter(Note.slug == candidate)
        if paper_id:
            query = query.filter(Note.paper_id == paper_id)
        else:
            query = query.filter(Note.paper_id.is_(None))
        if query.first() is None:
            return candidate
        candidate = f"{base}-{suffix}"
        suffix += 1
        if suffix > 500:
            # 极端情况：用随机后缀兜底，避免死循环
            from ..models.base import new_id

            return f"{base}-{new_id()[:6].lower()}"


def delete_note(note_id: str, *, remove_file: bool = False) -> None:
    """删除笔记。

    默认**不删磁盘文件**——用户在 Obsidian 里可能还指着它。
    要连文件一起删得显式传 ``remove_file=True``。
    """
    note = get_or_404(note_id)
    path = note.file_path

    db.session.delete(note)
    db.session.commit()

    if remove_file and path:
        import os

        try:
            os.remove(path)
            log.info("已删除笔记文件 %s", path)
        except OSError as exc:
            log.warning("删除笔记文件失败 %s：%s", path, exc)

    log.info("已删除笔记 %s", note_id)


def _try_write(note: Note) -> None:
    """尝试写盘，失败不抛异常。

    数据库是权威副本：写盘失败应该让用户在界面上看到「未同步」标记，
    而不是让整个编辑操作失败——那样用户会以为内容丢了，其实还在库里。
    """
    from .notesync import NoteSyncError, write_note

    try:
        write_note(note)
    except NoteSyncError as exc:
        note.sync_state = "error"
        note.sync_error = str(exc)
        db.session.commit()
        log.warning("笔记 %s 写盘失败：%s", note.id, exc)


# --------------------------------------------------------------------------
# 版本历史
# --------------------------------------------------------------------------


def list_revisions(note_id: str) -> list[NoteRevision]:
    return (
        db.session.query(NoteRevision)
        .filter(NoteRevision.note_id == note_id)
        .order_by(NoteRevision.version.desc())
        .all()
    )


def get_revision(note_id: str, version: int) -> NoteRevision | None:
    return (
        db.session.query(NoteRevision)
        .filter_by(note_id=note_id, version=version)
        .one_or_none()
    )


def restore_revision(note_id: str, version: int, *, write_disk: bool = True) -> Note:
    """回滚到某个历史版本。

    回滚本身也是一次修改——当前内容会先被快照，所以「回滚错了」
    还能再回滚回来。
    """
    note = get_or_404(note_id)
    revision = get_revision(note_id, version)
    if revision is None:
        raise NoteError(f"版本 {version} 不存在")

    _snapshot(note, author="system", summary=f"回滚到版本 {version} 之前")

    note.title = revision.title
    note.content_md = revision.content_md
    note.version += 1
    db.session.commit()

    if write_disk:
        _try_write(note)
    _reindex(note, {"content_md"})

    log.info("笔记 %s 已回滚到版本 %d", note_id, version)
    return note


def _content_at(note: Note, version: int) -> tuple[str, str]:
    """取出某个版本的内容，返回 ``(标题, 正文)``。

    **当前版本也要能取到。** 快照只在「内容即将被覆盖」时写入，
    所以 ``note.version`` 那一版永远不在 ``note_revisions`` 里——
    它的内容就是笔记本身。如果不做这个兜底，「我上次改了什么」这个
    最常用的对比就会报「版本不存在」，而那正是用户最想看的 diff。
    """
    if version == note.version:
        return note.title, note.content_md

    revision = get_revision(note.id, version)
    if revision is None:
        raise NoteError(f"版本 {version} 不存在")
    return revision.title, revision.content_md


def diff_revisions(note_id: str, from_version: int, to_version: int | None = None) -> str:
    """两个版本之间的统一 diff 文本。

    ``to_version`` 省略时对比「指定版本 → 当前内容」，
    这是界面上最常用的用法。
    """
    import difflib

    note = get_or_404(note_id)
    if to_version is None:
        to_version = note.version

    _, left = _content_at(note, from_version)
    _, right = _content_at(note, to_version)

    # splitlines() 去掉行尾 + lineterm="" + 用 \n 连接，三者必须配套：
    # 用 splitlines(keepends=True) 再配 lineterm="" 的话，头几行（---/+++/@@）
    # 不带换行符而内容行带着，拼出来会糊成一整行。
    lines = difflib.unified_diff(
        left.splitlines(),
        right.splitlines(),
        fromfile=f"v{from_version}",
        tofile=f"v{to_version}" + ("（当前）" if to_version == note.version else ""),
        lineterm="",
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------


def export_markdown(note_id: str) -> str:
    """导出为带 frontmatter 的 Markdown 文本。"""
    from .notesync import render_markdown

    note = get_or_404(note_id)
    paper = db.session.get(Paper, note.paper_id) if note.paper_id else None
    tags = [link.tag.name for link in note.tag_links if link.tag]
    return render_markdown(note, paper, tags=tags)


def to_dict(note: Note, *, include_content: bool = True) -> dict:
    from ..utils.time import iso

    data: dict[str, Any] = {
        "id": note.id,
        "paper_id": note.paper_id,
        "title": note.title,
        "kind": note.kind,
        "status": note.status,
        "source": note.source,
        "version": note.version,
        "sync_state": note.sync_state,
        "sync_error": note.sync_error,
        "file_path": note.file_path,
        "model": note.model,
        "created_at": iso(note.created_at),
        "updated_at": iso(note.updated_at),
        "tags": [
            {"id": link.tag_id, "name": link.tag.name if link.tag else None}
            for link in note.tag_links
        ],
    }
    if include_content:
        data["content_md"] = note.content_md
    return data


def stats() -> dict:
    total = db.session.query(Note).count()
    drafts = db.session.query(Note).filter(Note.status == STATUS_DRAFT).count()
    unsynced = db.session.query(Note).filter(Note.sync_state != "synced").count()
    by_kind: dict[str, int] = {}
    for kind, count in (
        db.session.query(Note.kind, db.func.count(Note.id)).group_by(Note.kind).all()
    ):
        by_kind[kind] = count
    return {"total": total, "drafts": drafts, "unsynced": unsynced, "by_kind": by_kind}


__all__ = [
    "NoteError",
    "create_note",
    "delete_note",
    "diff_revisions",
    "export_markdown",
    "find_by_kb_id",
    "get_note",
    "list_notes",
    "list_revisions",
    "restore_revision",
    "stats",
    "to_dict",
    "update_note",
]
