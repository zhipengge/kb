"""笔记与磁盘 Markdown 的双向同步。

**这是全系统最容易出错的地方**，因为同一个文件有两个写入方：本应用，和用户
（用 Obsidian、VS Code、任何编辑器）。文件系统不会告诉我们「是谁改的」，
所以必须用「预期值比对」来推断（见 ``detect_external_change``）。

三条规则贯穿全文：

**一、数据库是权威副本。** 网页端编辑能精确控制写入时机与内容，外部编辑不能。
但这不意味着可以拿库里覆盖磁盘——那会让用户在 Obsidian 里写的东西凭空消失。

**二、冲突不自动解决。** 发现两侧都变过，就把两边都留成文件，让用户选。
自动合并看起来很聪明，但合错了是静默丢内容，代价远高于多点一下。

**三、身份写在文件里。** frontmatter 里的 ``kb_id`` 是笔记的真实身份。
用户在 Obsidian 里重命名文件、挪目录，靠它仍能认回同一条记录——
这比任何基于路径或时间戳的推断都可靠。

写入用「临时文件 + os.replace」：os.replace 在同一文件系统上是原子的，
所以不会出现「写到一半断电，笔记变成半截」的情况。这一点对知识库很重要，
因为它是唯一无法从别处恢复的数据。
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..extensions import db
from ..models import Chunk, Note, Paper
from ..models.chunk import CHUNK_FIGURE
from .paths import normalize, sanitize_filename, slugify

log = logging.getLogger(__name__)

FRONTMATTER_FENCE = "---"

# 冲突副本放进这个子目录，而不是和正常笔记混在一起
CONFLICT_DIR = "_conflicts"
ASSETS_DIR = "_assets"


class NoteSyncError(RuntimeError):
    """同步失败。消息面向用户。"""


# --------------------------------------------------------------------------
# Markdown 与 frontmatter
# --------------------------------------------------------------------------


def render_markdown(
    note: Note,
    paper: Paper | None = None,
    *,
    tags: list[str] | None = None,
    line_ending: str = "lf",
) -> str:
    """把笔记渲染成带 YAML frontmatter 的 Markdown。

    frontmatter 的字段刻意保持精简且兼容 Obsidian：``tags`` 用列表形式，
    ``aliases`` 留给用户自己加。``kb_id`` 是本应用专用的字段，
    Obsidian 会忽略不认识的前置元数据，不会造成干扰。
    """
    front: dict[str, Any] = {
        "kb_id": note.id,
        "title": note.title or "",
    }

    if paper is not None:
        front["paper"] = paper.title or ""
        front["paper_id"] = paper.id
        if paper.arxiv_id:
            front["arxiv"] = paper.arxiv_id
        if paper.doi:
            front["doi"] = paper.doi
        if paper.year:
            front["year"] = paper.year

    front["kind"] = note.kind
    front["status"] = note.status
    front["created"] = note.created_at.isoformat() if note.created_at else None
    front["updated"] = note.updated_at.isoformat() if note.updated_at else None

    if note.model:
        front["model"] = note.model
    if note.prompt_version:
        front["prompt_version"] = note.prompt_version
    if tags:
        front["tags"] = sorted(tags)

    # allow_unicode 让中文标题保持可读；sort_keys=False 保持插入顺序，
    # 这样 diff 里字段顺序稳定，git 历史更干净
    header = yaml.safe_dump(
        {k: v for k, v in front.items() if v is not None},
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()

    body = (note.content_md or "").rstrip() + "\n"
    text = f"{FRONTMATTER_FENCE}\n{header}\n{FRONTMATTER_FENCE}\n\n{body}"

    if line_ending == "crlf":
        text = text.replace("\n", "\r\n")
    return text


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)


def parse_markdown(text: str) -> tuple[dict[str, Any], str]:
    """拆出 frontmatter 与正文。

    没有 frontmatter 时返回空字典与全文——用户可能新建了一个纯文本文件，
    这种情况应该被当作「新笔记」而不是报错。
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        log.warning("frontmatter 解析失败：%s", exc)
        return {}, text

    if not isinstance(meta, dict):
        return {}, text

    body = text[match.end():]
    return meta, body


def content_hash(text: str) -> str:
    """笔记内容哈希，用于判断文件是否被改动过。

    计算前要做两件归一化，两者都是必须的：

    **一、换行符。** 同一份内容在 Windows（CRLF）和 WSL（LF）下应该算出
    同一个值，否则两端编辑同一个文件会永远判定为「有冲突」。

    **二、剔除 ``updated`` 字段。** 这个字段来自 ``note.updated_at``，
    而它带 ``onupdate=utcnow``——也就是说**任何一次 ORM 提交都会刷新它**，
    哪怕内容一个字没改。若把它算进哈希，「刚写完文件」和「把库里内容渲染
    出来」会得到不同的哈希，三方比较直接失去意义（实测表现为：刚写完的
    笔记被判定成需要回写）。

    所以哈希代表的是「实质内容」，时间戳不参与。这也符合直觉：
    一篇笔记被 touch 一下不算改动。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _strip_volatile_frontmatter(normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _strip_volatile_frontmatter(text: str) -> str:
    """去掉 frontmatter 里不参与内容比对的字段。

    只在前置元数据区内替换，避免误伤正文里恰好以 ``updated:`` 开头的行。
    """
    if not text.startswith(FRONTMATTER_FENCE):
        return text

    end = text.find(f"\n{FRONTMATTER_FENCE}", len(FRONTMATTER_FENCE))
    if end < 0:
        return text

    head = text[:end]
    tail = text[end:]
    head = re.sub(r"^updated:.*$", "", head, flags=re.M)
    return head + tail


# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------


def note_path(note: Note, paper: Paper | None, *, notes_root: str, template: str) -> Path:
    """计算笔记应该落在哪个文件。

    目录模板支持 ``{year}`` ``{paper_slug}`` ``{paper_title}`` ``{kind}``。
    文件名用 ``<slug>.md``，slug 里保留了 CJK 字符——中文标题变成
    ``untitled-1`` 之类会让人完全找不到文件。
    """
    root = normalize(notes_root)

    paper_slug = "unfiled"
    year = "unknown"
    paper_title = ""
    if paper is not None:
        paper_slug = slugify(paper.title or paper.id, fallback=paper.id[:12])
        year = str(paper.year or "unknown")
        paper_title = slugify(paper.title or "", fallback="paper")

    try:
        subdir = template.format(
            year=year, paper_slug=paper_slug, paper_title=paper_title, kind=note.kind
        )
    except (KeyError, IndexError, ValueError) as exc:
        log.warning("笔记目录模板 %r 无效（%s），改用默认结构", template, exc)
        subdir = f"{year}/{paper_slug}"

    name = sanitize_filename(note.slug or note.title or note.id, fallback=note.id[:12])
    if not name.endswith(".md"):
        name += ".md"

    return root / subdir / name


def _atomic_write(path: Path, text: str) -> int:
    """原子写入，返回写入后的 mtime（纳秒）。

    先写临时文件再 os.replace：同目录下的 rename 是原子操作，
    读者要么看到旧内容要么看到新内容，不会看到写了一半的文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())  # 确保数据真的落盘，而不是只到了页缓存
        os.replace(tmp_name, path)
    except Exception:
        # 清理临时文件。它本身失败无所谓——真正的问题在上面，
        # 让异常带着原始原因抛出去，不要被清理失败盖掉。
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise

    return path.stat().st_mtime_ns


# --------------------------------------------------------------------------
# 写入
# --------------------------------------------------------------------------


def write_note(note: Note, *, force: bool = False) -> str:
    """把笔记写到磁盘。

    ``force=False`` 时，如果检测到外部改动会拒绝写入并抛错——
    调用方应该先解决冲突。这是「不静默覆盖用户内容」的执行点。
    """
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    if not settings.get("notes.write_to_disk"):
        return ""

    paper = db.session.get(Paper, note.paper_id) if note.paper_id else None
    tags = [link.tag.name for link in note.tag_links if link.tag]

    text = render_markdown(
        note, paper, tags=tags, line_ending=settings.get("notes.line_ending") or "lf"
    )
    digest = content_hash(text)

    target = Path(note.file_path) if note.file_path else note_path(
        note,
        paper,
        notes_root=settings.notes_root,
        template=settings.get("notes.dir_template"),
    )

    # 已存在且内容一致：什么都不做，避免无意义的 mtime 变化
    # （mtime 一变，Obsidian 之类的工具会认为文件被改动了）
    if target.is_file() and not force:
        existing = target.read_text(encoding="utf-8", errors="replace")
        if content_hash(existing) == digest:
            note.file_path = str(target)
            note.file_hash = digest
            note.written_mtime_ns = target.stat().st_mtime_ns
            note.sync_state = "synced"
            db.session.commit()
            return str(target)

    try:
        mtime_ns = _atomic_write(target, text)
    except OSError as exc:
        note.sync_state = "error"
        note.sync_error = str(exc)
        db.session.commit()
        raise NoteSyncError(f"写入笔记文件失败：{exc}") from exc

    note.file_path = str(target)
    note.file_hash = digest
    # 记下「我们写入后文件应该是什么 mtime」。下次发现磁盘 mtime 与此不同，
    # 就说明有人在我们之后动过这个文件——这是检测外部编辑的唯一依据。
    note.written_mtime_ns = mtime_ns
    note.sync_state = "synced"
    note.sync_error = None
    db.session.commit()

    _write_assets(note, paper)
    return str(target)


def _write_assets(note: Note, paper: Paper | None) -> None:
    """把笔记引用的图片从论文目录复制到 assets 下。

    目前只在笔记确实引用了论文插图时才复制。这样做而不是在索引阶段
    一次性导出全部图片，是为了避免几百篇论文的插图把笔记目录撑爆。
    """
    if paper is None or not note.file_path:
        return

    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    root = normalize(settings.notes_root)
    paper_slug = slugify(paper.title or paper.id, fallback=paper.id[:12])
    assets = root / ASSETS_DIR / paper_slug

    # 找出笔记里形如 ![](xxx.png) 的引用
    referenced = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", note.content_md or "")
    if not referenced:
        return

    # 插图由解析阶段导出到数据目录，这里只做映射与复制，不重新渲染 PDF
    figures = (
        db.session.query(Chunk).filter_by(paper_id=paper.id, kind=CHUNK_FIGURE).all()
    )
    by_name = {Path(c.meta.get("asset", "")).name: c for c in figures if c.meta}

    for reference in referenced:
        name = Path(reference).name
        chunk = by_name.get(name)
        if chunk is None:
            continue
        source = chunk.meta.get("asset_path")
        if not source or not os.path.isfile(source):
            continue

        assets.mkdir(parents=True, exist_ok=True)
        destination = assets / name
        if destination.exists():
            continue
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            log.warning("复制插图失败 %s：%s", source, exc)


# --------------------------------------------------------------------------
# 变更检测
# --------------------------------------------------------------------------


def library_hash(note: Note) -> str:
    """算出「把库里当前内容写出去会得到什么」的哈希。

    这是三路比较里代表「库这一侧」的那一路。没有它就无法区分
    「磁盘被改了」和「库被改了但还没写出」——只看磁盘与 ``note.file_hash``
    （上次写出的内容）的话，这两种情况长得一模一样。
    """
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    paper = db.session.get(Paper, note.paper_id) if note.paper_id else None
    tags = [link.tag.name for link in note.tag_links if link.tag]
    text = render_markdown(
        note, paper, tags=tags, line_ending=settings.get("notes.line_ending") or "lf"
    )
    return content_hash(text)


def detect_external_change(note: Note) -> str:
    """三方比对，判断磁盘文件相对数据库处于什么状态。

    三个输入：

      ``disk_hash``     磁盘上现在是什么
      ``library_hash``  库里现在是什么（写出去会是什么样）
      ``note.file_hash``上次我们写出去的是什么 —— **基线**

    返回值：

      ``missing``   文件不在了（被删除或移动）
      ``clean``     两侧一致
      ``modified``  只有磁盘变了 → 可以安全导入
      ``dirty``     只有库变了 → 把库里的内容写出去即可
      ``conflict``  两侧都变了 → 必须由人来决定
    """
    if not note.file_path:
        return "missing"

    path = Path(note.file_path)
    if not path.is_file():
        return "missing"

    try:
        on_disk = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("读取笔记文件失败 %s：%s", path, exc)
        return "missing"

    disk_hash = content_hash(on_disk)
    current_hash = library_hash(note)

    # 两侧内容一致，就是同步的。mtime 变过（比如被 touch）不算改动。
    if disk_hash == current_hash:
        return "clean"

    baseline = note.file_hash or ""

    # 没有基线（例如这条记录是外部导入进来的，我们从没写过文件）：
    # 谈不上「谁改了」，把磁盘内容当作待导入的来源。
    if not baseline:
        return "modified"

    disk_changed = disk_hash != baseline
    library_changed = current_hash != baseline

    if disk_changed and library_changed:
        return "conflict"
    if disk_changed:
        return "modified"
    return "dirty"


def import_from_disk(note: Note) -> Note:
    """把磁盘上的内容读回数据库。

    只应在 ``detect_external_change`` 判定为 ``modified`` 时调用——
    也就是说，确认库里那侧没有更新的改动会丢失。
    """
    if not note.file_path:
        raise NoteSyncError("笔记没有关联文件")

    path = Path(note.file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    meta, body = parse_markdown(text)

    if body.strip() != (note.content_md or "").strip():
        _snapshot(note, author="external", summary="导入磁盘上的外部改动")

    note.content_md = body.rstrip()
    if meta.get("title"):
        note.title = str(meta["title"])
    note.file_hash = content_hash(text)
    note.written_mtime_ns = path.stat().st_mtime_ns
    note.sync_state = "synced"
    note.sync_error = None
    note.version += 1
    db.session.commit()

    _sync_tags_from_frontmatter(note, meta.get("tags") or [])
    # 外部改动也要重建索引：用户在 Obsidian 里改了内容，检索必须跟着变。
    # 漏掉这一步的表现是「改了但搜不到」，而且不会有任何提示。
    from .indexer import reindex_note_safely

    reindex_note_safely(note)
    log.info("已从磁盘导入笔记 %s 的外部改动", note.id)
    return note


def _sync_tags_from_frontmatter(note: Note, names: list) -> None:
    """把 frontmatter 里的 tags 同步成标签关联。

    只增不减：用户在 Obsidian 里加的标签会被采纳，但不移除库里的标签——
    frontmatter 可能被 Obsidian 插件重写过而丢字段，据此删标签风险太大。
    """
    from .tagging import attach_tag

    for name in names:
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            attach_tag(note, name.strip(), source="import")
        except Exception:
            log.warning("导入标签 %r 失败", name, exc_info=True)


def make_conflict_copy(note: Note) -> Path | None:
    """把磁盘上的版本另存为冲突副本。

    命名带时间戳而不是 ``.orig``：同一条笔记可能反复冲突，
    固定后缀会让第二次直接覆盖掉第一次的副本。
    """
    if not note.file_path:
        return None

    source = Path(note.file_path)
    if not source.is_file():
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = source.parent / CONFLICT_DIR / f"{source.stem}.conflict-{stamp}.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(source, target)
    except OSError as exc:
        log.error("创建冲突副本失败：%s", exc)
        return None

    log.warning("笔记 %s 存在冲突，磁盘版本已另存为 %s", note.id, target)
    return target


# --------------------------------------------------------------------------
# 批量操作
# --------------------------------------------------------------------------


def sync_all(ctx=None, *, pull_only: bool = False) -> dict:
    """扫描所有笔记的同步状态。

    ``pull_only=True`` 时只把外部改动读回来，不写任何文件——
    适合「我在 Obsidian 里改了一堆，先同步进来」的场景。
    """
    notes = db.session.query(Note).filter(Note.file_path.isnot(None)).all()
    result: dict = {
        "total": len(notes),
        "clean": 0, "modified": 0, "conflict": 0, "missing": 0, "dirty": 0,
        "imported": 0, "written": 0, "errors": [],
    }

    for index, note in enumerate(notes):
        if ctx is not None and index % 20 == 0:
            ctx.check_cancelled()
            ctx.progress(index / max(1, len(notes)), f"检查 {index + 1}/{len(notes)}")

        state = detect_external_change(note)
        result[state] = result.get(state, 0) + 1

        try:
            if state == "modified":
                # 磁盘被改过，而库这侧自上次写出以来没动过 —— 导入是安全的
                import_from_disk(note)
                result["imported"] += 1

            elif state == "dirty":
                # 只有库这侧变了（例如网页端保存时写盘失败）。把库里的内容
                # 补写到磁盘即可，不涉及任何内容取舍。
                if not pull_only:
                    write_note(note, force=True)
                    result["written"] += 1

            elif state == "conflict":
                # 两侧都变过：**不自动选边**。把磁盘版本另存一份，
                # 库里保留自己那份，由用户决定最终留哪个。
                copy = make_conflict_copy(note)
                note.sync_state = "conflict"
                note.sync_error = f"磁盘与库内容不一致，磁盘版本已另存为 {copy}"
                db.session.commit()

            elif state == "missing":
                note.sync_state = "missing"
                note.sync_error = "文件已不在磁盘上"
                db.session.commit()

            elif state == "clean" and note.sync_state != "synced":
                note.sync_state = "synced"
                note.sync_error = None
                db.session.commit()

        except Exception as exc:
            result["errors"].append(f"{note.id}: {exc}")
            log.exception("同步笔记 %s 失败", note.id)

    # 库里还没落盘的笔记，补写出去
    if not pull_only:
        pending = db.session.query(Note).filter(Note.file_path.is_(None)).all()
        for note in pending:
            if ctx is not None:
                ctx.check_cancelled()
            try:
                write_note(note)
                result["written"] += 1
            except Exception as exc:
                result["errors"].append(f"{note.id}: {exc}")

    if result["errors"]:
        db.session.rollback()

    log.info(
        "笔记同步：共 %d 篇，一致 %d，外部改动 %d，冲突 %d，缺失 %d",
        result["total"], result["clean"], result["modified"],
        result["conflict"], result["missing"],
    )
    return result


def find_untracked_files() -> list[dict]:
    """找出笔记目录里存在、但数据库里没有对应记录的 .md 文件。

    这是「用户直接在 Obsidian 里新建笔记」的入口。不自动导入——
    自动导入会产生一堆来源不明的记录，让用户先看一眼更稳妥。
    """
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    notes_root = settings.notes_root
    if not notes_root:
        return []

    try:
        root = normalize(notes_root)
    except (OSError, ValueError):
        return []
    if not root.is_dir():
        return []

    tracked = {
        str(Path(path).resolve())
        for (path,) in db.session.query(Note.file_path).filter(Note.file_path.isnot(None)).all()
    }

    untracked: list[dict] = []
    for path in root.rglob("*.md"):
        if CONFLICT_DIR in path.parts or ASSETS_DIR in path.parts:
            continue
        try:
            resolved = str(path.resolve())
        except OSError:
            continue
        if resolved in tracked:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        meta, body = parse_markdown(text)

        untracked.append(
            {
                "path": str(path),
                "title": meta.get("title") or path.stem,
                "kb_id": meta.get("kb_id"),
                "size": len(text),
                "preview": body.strip()[:160],
                "has_kb_id": bool(meta.get("kb_id")),
            }
        )
        if len(untracked) >= 200:
            break

    return untracked


def _snapshot(note: Note, *, author: str, summary: str) -> None:
    """写入历史版本。由 notes 服务复用。"""
    from ..models import NoteRevision

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
            summary=summary,
        )
    )


__all__ = [
    "ASSETS_DIR",
    "CONFLICT_DIR",
    "NoteSyncError",
    "content_hash",
    "detect_external_change",
    "find_untracked_files",
    "import_from_disk",
    "make_conflict_copy",
    "note_path",
    "parse_markdown",
    "render_markdown",
    "sync_all",
    "write_note",
]
