"""目录扫描：发现论文、识别变更、去重。

这是整个系统里唯一需要处理「用户绕过程序直接改磁盘」的地方，所以逻辑比看上去复杂。
四种变更都要能正确识别：

  新增      磁盘上有、库里没有         -> 建记录
  修改      路径相同、内容变了         -> 更新哈希与元数据，标记需重新解析
  移动/改名 内容相同、路径变了         -> 更新路径，**不**新建记录
  删除      库里有、磁盘上没有         -> 软删除

其中「移动」最容易做错：只按路径比对的实现会把它当成「删除 + 新增」，
于是论文拿到一个新 id，挂在它上面的笔记、标签、阅读进度全部失联。
这里的做法是：内容哈希相同就认为是同一个文件换了地方。

性能上有一条关键约束——**不能每次都重新哈希所有文件**。5 万个 PDF 全量
sha256 要跑很久，而磁盘 IO 是这里最贵的资源。所以先比 (路径, mtime, size)，
三者都没变就直接跳过，哈希只对新增和变更的文件计算。
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..extensions import db
from ..models import ScanRun
from ..models.base import utcnow
from ..models.paper import (
    INGEST_PENDING,
    SOURCE_LOCAL,
    Paper,
    PaperDuplicate,
)
from .paths import PathError, is_excluded, normalize, path_key, validate_root

log = logging.getLogger(__name__)

_CHUNK = 1024 * 1024  # 1 MiB：大文件分块读，避免一次性读进内存


def file_digest(path: str | Path) -> str:
    """计算文件内容的 sha256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ScanStats:
    """一次扫描的结果。会原样进 API 响应与任务日志。"""

    files_seen: int = 0
    files_added: int = 0
    files_moved: int = 0
    files_changed: int = 0
    files_missing: int = 0
    files_skipped: int = 0
    files_unreadable: int = 0
    duplicates_found: int = 0
    hashes_computed: int = 0
    roots_scanned: int = 0
    roots_missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "files_seen": self.files_seen,
            "files_added": self.files_added,
            "files_moved": self.files_moved,
            "files_changed": self.files_changed,
            "files_missing": self.files_missing,
            "files_skipped": self.files_skipped,
            "files_unreadable": self.files_unreadable,
            "duplicates_found": self.duplicates_found,
            "hashes_computed": self.hashes_computed,
            "roots_scanned": self.roots_scanned,
            "roots_missing": self.roots_missing,
            "errors": self.errors[:50],
            "elapsed_ms": self.elapsed_ms,
        }


# --------------------------------------------------------------------------
# 遍历
# --------------------------------------------------------------------------


def iter_pdf_files(
    root: Path,
    *,
    max_depth: int = 12,
    follow_symlinks: bool = False,
    ignore_hidden: bool = True,
    exclude: list[str] | None = None,
):
    """递归遍历目录，产出 PDF 文件路径。

    用手写的栈而不是 ``os.walk``：需要精确控制深度、符号链接与排除规则，
    而 ``os.walk`` 的语义（尤其是 followlinks 与 prune 的配合）不够直观。
    另外手写之后可以在遍历中途检查取消信号。
    """
    stack: list[tuple[Path, int]] = [(root, 0)]
    seen_dirs: set[str] = set()

    while stack:
        directory, depth = stack.pop()
        if depth > max_depth:
            log.debug("达到最大深度，跳过：%s", directory)
            continue

        try:
            # 用 scandir 而不是 listdir：在 9p/网络挂载上，scandir 能少一次
            # stat 系统调用，大目录下差别很明显
            entries = list(os.scandir(directory))
        except PermissionError:
            log.warning("无权限读取目录：%s", directory)
            continue
        except OSError as exc:
            log.warning("读取目录失败 %s：%s", directory, exc)
            continue

        for entry in entries:
            name = entry.name

            if ignore_hidden and name.startswith("."):
                continue
            if is_excluded(name, exclude):
                continue

            try:
                is_dir = entry.is_dir(follow_symlinks=follow_symlinks)
                is_file = entry.is_file(follow_symlinks=follow_symlinks)
            except OSError:
                continue

            if is_dir:
                if entry.is_symlink() and not follow_symlinks:
                    continue
                # 防止符号链接造成目录环
                try:
                    real = os.path.realpath(entry.path)
                except OSError:
                    continue
                if real in seen_dirs:
                    continue
                seen_dirs.add(real)
                stack.append((Path(entry.path), depth + 1))

            elif is_file and name.lower().endswith(".pdf"):
                yield Path(entry.path)


# --------------------------------------------------------------------------
# 去重
# --------------------------------------------------------------------------


def _record_duplicate(paper: Paper, candidate: Paper, reason: str, score: float, detail: dict) -> bool:
    """记录一条重复候选。已存在则跳过（保持幂等）。"""
    if paper.id == candidate.id:
        return False

    # 统一顺序，避免 (A,B) 和 (B,A) 各存一条
    left, right = sorted([paper.id, candidate.id])
    exists = (
        db.session.query(PaperDuplicate)
        .filter_by(paper_id=left, candidate_id=right, reason=reason)
        .one_or_none()
    )
    if exists is not None:
        return False

    db.session.add(
        PaperDuplicate(
            paper_id=left,
            candidate_id=right,
            reason=reason,
            score=score,
            detail=detail,
            status="pending",
        )
    )
    return True


def _find_fuzzy_duplicates(paper: Paper, threshold: float) -> list[Paper]:
    """找标题相似的其它论文。

    用 rapidfuzz 的 token_set_ratio 而不是简单的编辑距离：论文标题常有
    副标题、大小写、标点的差异，token 级别的集合比较对这些更宽容。
    """
    if not paper.title_norm or len(paper.title_norm) < 8:
        return []

    from rapidfuzz import fuzz

    # 只在年份相近的候选里比——年份差太多的标题相似基本是巧合
    query = db.session.query(Paper).filter(
        Paper.id != paper.id,
        Paper.deleted_at.is_(None),
    )
    if paper.year:
        query = query.filter(Paper.year.isnot(None), Paper.year.between(paper.year - 2, paper.year + 2))

    matches: list[tuple[float, Paper]] = []
    for other in query.limit(500):
        if not other.title_norm:
            continue
        score = fuzz.token_set_ratio(paper.title_norm, other.title_norm)
        if score >= threshold:
            matches.append((score, other))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in matches[:5]]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def scan_roots(ctx=None, roots: list[str] | None = None, full: bool = False) -> dict:
    """扫描配置的论文根目录。

    ``ctx`` 是 ``JobContext``（可为 None，CLI 同步调用时不传）。
    ``full=True`` 时忽略 mtime/size 快路径，强制重新检查每个文件。
    """
    started = time.perf_counter()
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    stats = ScanStats()

    target_roots = roots if roots is not None else settings.papers_roots
    if not target_roots:
        stats.errors.append("没有配置任何论文根目录")
        return stats.to_dict()

    max_depth = int(settings.get("scan.max_depth"))
    follow_symlinks = bool(settings.get("scan.follow_symlinks"))
    ignore_hidden = bool(settings.get("scan.ignore_hidden"))
    exclude = settings.get("scan.exclude_globs") or []
    dedup_exact = bool(settings.get("scan.dedup_exact"))
    dedup_identifier = bool(settings.get("scan.dedup_identifier"))
    dedup_fuzzy = bool(settings.get("scan.dedup_fuzzy"))
    fuzzy_threshold = float(settings.get("scan.fuzzy_threshold"))

    def report(progress: float, message: str) -> None:
        if ctx is not None:
            ctx.progress(progress, message)
            ctx.heartbeat()

    # 本次扫描看到的路径，用于识别「消失」的文件
    seen_paths: set[str] = set()
    run_records: list[ScanRun] = []

    for root_index, raw_root in enumerate(target_roots):
        if ctx is not None:
            ctx.check_cancelled()

        try:
            root = validate_root(raw_root, must_exist=False)
        except PathError as exc:
            stats.errors.append(f"{raw_root}：{exc}")
            continue

        if not root.is_dir():
            stats.roots_missing.append(str(root))
            log.warning("根目录不存在，已跳过：%s", root)
            continue

        stats.roots_scanned += 1
        run = ScanRun(root=str(root), status="running")
        db.session.add(run)
        db.session.commit()
        run_records.append(run)

        root_index_base = root_index / len(target_roots)

        for pdf_path in iter_pdf_files(
            root,
            max_depth=max_depth,
            follow_symlinks=follow_symlinks,
            ignore_hidden=ignore_hidden,
            exclude=exclude,
        ):
            if ctx is not None and stats.files_seen % 25 == 0:
                ctx.check_cancelled()

            stats.files_seen += 1
            _handle_file(
                pdf_path=pdf_path,
                root=root,
                stats=stats,
                seen_paths=seen_paths,
                run=run,
                full=full,
                dedup_exact=dedup_exact,
                dedup_identifier=dedup_identifier,
                dedup_fuzzy=dedup_fuzzy,
                fuzzy_threshold=fuzzy_threshold,
            )

            if stats.files_seen % 20 == 0:
                report(
                    min(0.95, root_index_base + (stats.files_seen % 1000) / 20000),
                    f"已扫描 {stats.files_seen} 个文件（新增 {stats.files_added}）",
                )

        run.finished_at = utcnow()
        run.status = "succeeded"
        run.files_seen = stats.files_seen
        run.files_added = stats.files_added
        run.files_moved = stats.files_moved
        run.files_missing = stats.files_missing
        run.files_skipped = stats.files_skipped
        run.duplicates_found = stats.duplicates_found
        run.stats = stats.to_dict()
        db.session.commit()

    if ctx is not None:
        ctx.check_cancelled()

    # 识别消失的文件（软删除）
    if target_roots:
        _mark_missing(target_roots, seen_paths, stats)

    stats.elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "扫描完成：检查 %d，新增 %d，移动 %d，变更 %d，缺失 %d，跳过 %d（计算了 %d 个哈希，用时 %.1fs）",
        stats.files_seen, stats.files_added, stats.files_moved, stats.files_changed,
        stats.files_missing, stats.files_skipped, stats.hashes_computed,
        stats.elapsed_ms / 1000,
    )
    return stats.to_dict()


def _handle_file(
    *,
    pdf_path: Path,
    root: Path,
    stats: ScanStats,
    seen_paths: set[str],
    run: ScanRun,
    full: bool,
    dedup_exact: bool,
    dedup_identifier: bool,
    dedup_fuzzy: bool,
    fuzzy_threshold: float,
) -> None:
    """处理单个文件。所有分支都在这里，主循环保持可读。"""
    try:
        abs_path = str(pdf_path.absolute())
        key = path_key(abs_path)
        stat = pdf_path.stat()
    except OSError as exc:
        stats.files_unreadable += 1
        log.warning("无法访问 %s：%s", pdf_path, exc)
        return

    seen_paths.add(key)

    existing = db.session.query(Paper).filter(Paper.path_key == key).one_or_none()

    # ---- 快路径：路径相同且 mtime/size 未变 ----
    if existing is not None and not full:
        same_mtime = existing.file_mtime_ns == stat.st_mtime_ns
        same_size = existing.file_size == stat.st_size
        if same_mtime and same_size and existing.file_hash and existing.deleted_at is None:
            stats.files_skipped += 1
            return
        # 文件被改过，需要重算哈希
    elif existing is not None and existing.deleted_at is not None:
        # 之前被标记为「消失」的文件又回来了
        existing.deleted_at = None
        log.info("文件重新出现：%s", abs_path)

    # ---- 计算内容哈希 ----
    digest = None
    if dedup_exact:
        try:
            digest = file_digest(pdf_path)
            stats.hashes_computed += 1
        except OSError as exc:
            stats.files_unreadable += 1
            log.warning("计算哈希失败 %s：%s", pdf_path, exc)
            return

    # ---- 内容未变（哈希相同）----
    if existing is not None and digest and existing.file_hash == digest:
        existing.file_size = stat.st_size
        existing.file_mtime_ns = stat.st_mtime_ns
        existing.file_path = abs_path
        db.session.commit()
        stats.files_skipped += 1
        return

    # ---- 内容相同但路径不同 => 移动 或 重复 ----
    if digest:
        # 同一份内容在库里可能对应多条记录（原件 + 已知的重复副本），
        # 要挑出「最像是被移动的那一条」。
        #
        # 判据是**它的旧路径是否还存在**：旧路径消失了，说明就是它被挪走了；
        # 旧路径还在，说明磁盘上确实有两份，那是重复而不是移动。
        #
        # 不能只按 created_at 排序来挑：SQLite 的 CURRENT_TIMESTAMP 只有秒级精度，
        # 同一批扫描创建的记录时间戳完全相同，排序结果是不确定的。
        # 所以先取回全部候选，再在 Python 里按「旧路径是否存续」判断。
        twins = (
            db.session.query(Paper)
            .filter(Paper.file_hash == digest, Paper.path_key != key)
            .order_by(Paper.id)
            .all()
        )

        moved_candidates = [
            p for p in twins
            if p.deleted_at is None and p.file_path and not os.path.isfile(p.file_path)
        ]
        live_twins = [
            p for p in twins
            if p.deleted_at is None and p.file_path and os.path.isfile(p.file_path)
        ]

        if existing is None and moved_candidates:
            # 旧路径已经不在了 -> 这是一个移动/改名
            moved = moved_candidates[0]
            old_path = moved.file_path
            moved.file_path = abs_path
            moved.path_key = key
            moved.file_mtime_ns = stat.st_mtime_ns
            moved.file_size = stat.st_size
            moved.deleted_at = None
            db.session.commit()
            stats.files_moved += 1
            run.files_moved = stats.files_moved
            log.info("识别为移动（保留原记录与笔记）：%s -> %s", old_path, abs_path)
            return

        if existing is None and live_twins:
            # 两份都在 -> 真的重复。
            #
            # 这里刻意**仍然建一条新记录**，而不是直接跳过：如果只是跳过，
            # 用户就完全看不到磁盘上有两个一模一样的文件，也就无从清理。
            # 新记录带着元数据但不触发解析，并在 paper_duplicates 里挂一条
            # 待确认的候选，界面上可以合并（保留其一）或删除。
            twin = live_twins[0]
            shadow = Paper(
                title=twin.title,
                title_norm=twin.title_norm,
                authors=twin.authors,
                year=twin.year,
                venue=twin.venue,
                doi=twin.doi,
                arxiv_id=twin.arxiv_id,
                file_path=abs_path,
                path_key=key,
                file_hash=digest,
                file_size=stat.st_size,
                file_mtime_ns=stat.st_mtime_ns,
                source=SOURCE_LOCAL,
                ingest_status=INGEST_PENDING,
            )
            db.session.add(shadow)
            db.session.flush()
            if _record_duplicate(shadow, twin, "exact_hash", 1.0, {"path": abs_path}):
                stats.duplicates_found += 1
            db.session.commit()
            log.info("发现重复文件：%s 与 %s 内容相同", abs_path, twin.file_path)
            return

    # ---- 新增 ----
    if existing is None:
        paper = Paper(
            file_path=abs_path,
            path_key=key,
            file_hash=digest,
            file_size=stat.st_size,
            file_mtime_ns=stat.st_mtime_ns,
            source=SOURCE_LOCAL,
            ingest_status=INGEST_PENDING,
        )
        db.session.add(paper)
        db.session.flush()

        _fill_metadata(paper, pdf_path)

        if dedup_identifier and (paper.doi or paper.arxiv_id):
            _check_identifier_duplicate(paper, stats)

        if dedup_fuzzy:
            for other in _find_fuzzy_duplicates(paper, fuzzy_threshold):
                if _record_duplicate(paper, other, "title_fuzzy", fuzzy_threshold,
                                     {"title": paper.title, "other_title": other.title}):
                    stats.duplicates_found += 1

        db.session.commit()
        stats.files_added += 1
        run.files_added = stats.files_added

    # ---- 已存在但内容变了 ----
    else:
        existing.file_hash = digest
        existing.file_size = stat.st_size
        existing.file_mtime_ns = stat.st_mtime_ns
        existing.file_path = abs_path
        # 内容变了，之前的解析与索引都失效了
        if existing.ingest_status != INGEST_PENDING:
            existing.ingest_status = INGEST_PENDING
        _fill_metadata(existing, pdf_path)
        db.session.commit()
        stats.files_changed += 1


def _fill_metadata(paper: Paper, pdf_path: Path) -> None:
    """读取 PDF 元数据填进论文记录。

    只有标题是空的、或来自不可靠的来源（文件名）时才重新提取——
    用户手工改过的标题不该被扫描覆盖掉。
    """
    from .pdf import PdfError, normalize_title, read_metadata

    needs_title = not paper.title or paper.meta is None or paper.meta.get("title_source") == "filename"
    needs_identifiers = not paper.doi and not paper.arxiv_id
    if not (needs_title or needs_identifiers):
        return

    try:
        meta = read_metadata(pdf_path, read_first_page=True)
    except PdfError as exc:
        log.warning("读取 PDF 元数据失败 %s：%s", pdf_path, exc)
        paper.meta = {**(paper.meta or {}), "metadata_error": str(exc)}
        if not paper.title:
            from .pdf import title_from_filename

            paper.title = title_from_filename(pdf_path)
            paper.title_norm = normalize_title(paper.title)
        return

    if needs_title and meta.title:
        paper.title = meta.title
        paper.title_norm = normalize_title(meta.title)
    elif not paper.title:
        paper.title = pdf_path.stem
        paper.title_norm = normalize_title(paper.title)

    if not paper.authors and meta.authors:
        paper.authors = meta.authors
    if not paper.year and meta.year:
        paper.year = meta.year
    if needs_identifiers:
        paper.doi = meta.doi
        paper.arxiv_id = meta.arxiv_id
    if not paper.abstract and meta.abstract:
        paper.abstract = meta.abstract
    if not paper.page_count:
        paper.page_count = meta.page_count

    paper.meta = {
        **(paper.meta or {}),
        "title_source": meta.title_source,
    }


def _check_identifier_duplicate(paper: Paper, stats: ScanStats) -> None:
    """DOI / arXiv 号相同视为同一篇。"""
    conditions = []
    if paper.doi:
        conditions.append(Paper.doi == paper.doi)
    if paper.arxiv_id:
        conditions.append(Paper.arxiv_id == paper.arxiv_id)
    if not conditions:
        return

    from sqlalchemy import or_

    # 按 id 排序而不是 created_at：ULID 前缀是毫秒时间戳，天然有序，
    # 且不会像 SQLite 的 CURRENT_TIMESTAMP 那样退化成秒级精度后出现并列。
    other = (
        db.session.query(Paper)
        .filter(or_(*conditions), Paper.id != paper.id)
        .order_by(Paper.id)
        .first()
    )
    if other is None:
        return

    reason = "doi" if paper.doi and other.doi == paper.doi else "arxiv"
    if _record_duplicate(paper, other, reason, 1.0,
                         {"doi": paper.doi, "arxiv": paper.arxiv_id}):
        stats.duplicates_found += 1


def _mark_missing(roots: list[str], seen_paths: set[str], stats: ScanStats) -> None:
    """把位于已扫描根目录下、但本次没见到的论文标记为已删除。

    只对**本次实际扫描过的根目录**生效。否则用户临时只扫一个子目录时，
    其它根目录下的论文会被误判为「消失」。
    """
    normalized_roots: list[str] = []
    for raw in roots:
        try:
            normalized_roots.append(str(normalize(raw)))
        except (OSError, ValueError):
            continue

    if not normalized_roots:
        return

    candidates = (
        db.session.query(Paper)
        .filter(Paper.deleted_at.is_(None), Paper.source == SOURCE_LOCAL)
        .all()
    )
    now = utcnow()
    for paper in candidates:
        if not any(paper.file_path.startswith(root) for root in normalized_roots):
            continue
        if path_key(paper.file_path) in seen_paths:
            continue
        paper.deleted_at = now
        stats.files_missing += 1

    if stats.files_missing:
        db.session.commit()
        log.info("标记 %d 篇论文的文件已不存在（记录与笔记保留）", stats.files_missing)


def scan_summary() -> dict[str, Any]:
    """库的整体状况，供界面展示。"""
    total = db.session.query(Paper).filter(Paper.deleted_at.is_(None)).count()
    deleted = db.session.query(Paper).filter(Paper.deleted_at.isnot(None)).count()
    pending = (
        db.session.query(Paper)
        .filter(Paper.deleted_at.is_(None), Paper.ingest_status == INGEST_PENDING)
        .count()
    )
    duplicates = (
        db.session.query(PaperDuplicate).filter(PaperDuplicate.status == "pending").count()
    )
    return {
        "total": total,
        "deleted": deleted,
        "pending_ingest": pending,
        "pending_duplicates": duplicates,
    }


__all__ = ["ScanStats", "file_digest", "iter_pdf_files", "scan_roots", "scan_summary"]
