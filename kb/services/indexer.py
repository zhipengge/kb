"""把 PDF 变成可检索的分块。

流程：抽取正文 -> 识别章节 -> 分块 -> 写库（触发器自动同步全文索引）。

有两个刻意的设计：

**重新索引是「先删后插」而不是「尽量增量更新」。** 分块边界会随参数变化
（改了块大小，或换了章节识别逻辑），增量对不上号，结果是索引里混着两套
切法的块。整篇重来一遍既简单又不会错，而单篇论文的解析只要一秒左右。

**失败要留下痕迹。** 解析失败时把错误写进 ``paper.error`` 并把状态标为
failed，而不是静默跳过——否则用户看到的是「这篇论文搜不到」，
却不知道为什么。
"""

from __future__ import annotations

import logging
import time

from ..extensions import db
from ..models import Chunk, Paper
from ..models.chunk import CHUNK_CODE, CHUNK_NOTE, CHUNK_TEXT
from ..models.paper import INGEST_FAILED, INGEST_INDEXED, INGEST_PARSED, INGEST_PENDING
from .chunker import RULES_VERSION, rules_meta

log = logging.getLogger(__name__)


def index_papers(ctx=None, paper_ids: list[str] | None = None, force: bool = False) -> dict:
    """批量建立索引。后台任务与 CLI 都走这里。"""
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    target_tokens = int(settings.get("chunk.size"))
    overlap = int(settings.get("chunk.overlap"))

    query = db.session.query(Paper).filter(Paper.deleted_at.is_(None))
    if paper_ids:
        query = query.filter(Paper.id.in_(paper_ids))
    elif not force:
        # 只处理还没索引过的。force 时不加这个条件，全部重来。
        query = query.filter(Paper.ingest_status.in_((INGEST_PENDING, INGEST_PARSED, INGEST_FAILED)))

    papers = query.order_by(Paper.id).all()
    total = len(papers)

    started = time.perf_counter()
    result = {"total": total, "indexed": 0, "failed": 0, "skipped": 0, "chunks": 0, "errors": []}

    if total == 0:
        log.info("没有需要索引的论文")
        return result

    for index, paper in enumerate(papers):
        if ctx is not None:
            ctx.check_cancelled()
            ctx.progress(index / total, f"正在索引 {index + 1}/{total}：{(paper.title or '')[:40]}")

        try:
            count = index_paper(paper, target_tokens=target_tokens, overlap=overlap)
            result["indexed"] += 1
            result["chunks"] += count
        except Exception as exc:
            result["failed"] += 1
            message = f"{type(exc).__name__}: {exc}"
            result["errors"].append(f"{(paper.title or paper.id)[:60]}: {message}")
            paper.ingest_status = INGEST_FAILED
            paper.error = message
            db.session.commit()
            log.exception("索引论文失败：%s", paper.file_path)

    # 索引完成后合并 FTS 内部段，提升后续检索速度
    if ctx is not None:
        ctx.progress(0.98, "优化全文索引…")
    try:
        from ..sqlite import load_sqlite_vec  # noqa: F401  (保持导入一致性)
        from .fts import optimize

        optimize(db.engine)
    except Exception:
        log.exception("优化全文索引失败（不影响功能）")

    result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    log.info(
        "索引完成：%d 篇，%d 个分块，失败 %d 篇，用时 %.1fs",
        result["indexed"], result["chunks"], result["failed"], result["elapsed_ms"] / 1000,
    )
    return result


def index_paper(paper: Paper, *, target_tokens: int = 800, overlap: int = 120) -> int:
    """解析并索引单篇论文。返回写入的分块数。

    正文来源的优先级：本地 LaTeX 源码 -> arXiv 下载的源码 -> PDF 解析。
    选择逻辑在 ``services/sources.py`` 里，这里只消费结果。
    """
    from .chunker import chunk_sections, estimate_tokens
    from .pdf import PdfError
    from .sources import acquire_source

    if not paper.file_path:
        raise PdfError("论文没有关联文件")

    source = acquire_source(paper)
    if not source.ok:
        raise PdfError(
            "；".join(source.errors) if source.errors else "无法解析正文"
        )

    # 补全扫描阶段没拿到的元数据。LaTeX 源码里的标题/作者比 PDF 元数据可靠得多
    _enrich_from_source(paper, source)

    drafts = chunk_sections(
        source.sections, target_tokens=target_tokens, overlap_tokens=overlap
    )
    if not drafts:
        raise PdfError("解析出的章节里没有可索引的正文")

    # 先删旧块。FTS 索引由触发器跟着删，不需要手动处理。
    #
    # **代码块要留着**（kind=code）：它们来自另一条流水线（services/code_index），
    # 删除条件如果只看「属于这篇论文且不是笔记块」，重建索引就会把代码索引
    # 一并抹掉。那是个很难发现的损失——检索悄悄少了一部分内容，
    # 没有任何报错，要等用户问「代码在哪」答不出来才会察觉。
    deleted = (
        db.session.query(Chunk)
        .filter(
            Chunk.paper_id == paper.id,
            Chunk.note_id.is_(None),
            Chunk.kind != CHUNK_CODE,
        )
        .delete(synchronize_session=False)
    )
    if deleted:
        log.debug("清除论文 %s 的 %d 个旧分块", paper.id, deleted)

    order = 0
    for draft in drafts:
        db.session.add(
            Chunk(
                paper_id=paper.id,
                kind=CHUNK_TEXT,
                section_path=draft.section_path,
                section_index=None,
                page_from=draft.page_from,
                page_to=draft.page_to,
                ord=order,
                text=draft.text,
                n_tokens=draft.n_tokens,
                is_section_start=draft.is_section_start,
                meta={
                    "section_title": draft.section_path,
                    # 记下来源：界面要靠它解释「为什么这条引用只有章节号没有页码」
                    "source": source.source_type,
                    # 分块规则版本，用来查「这块是不是旧规则建的」
                    **rules_meta(),
                },
            )
        )
        order += 1

    # 公式与图注作为独立分块写入。它们自带 kind，检索时可以按类型过滤
    # （「找这个指标的数值」只查 table，「找公式」只查 formula）。
    for extra in source.extra_chunks:
        db.session.add(
            Chunk(
                paper_id=paper.id,
                kind=extra.get("kind", CHUNK_TEXT),
                section_path=extra.get("section_path"),
                page_from=extra.get("page") or 1,
                page_to=extra.get("page") or 1,
                ord=order,
                text=extra.get("text", ""),
                n_tokens=estimate_tokens(extra.get("text", "")),
                meta={
                    **(extra.get("meta") or {}),
                    "section_title": extra.get("section_path"),
                    "source": source.source_type,
                    **rules_meta(),
                },
            )
        )
        order += 1

    paper.ingest_status = INGEST_INDEXED
    paper.error = None
    if source.page_count:
        paper.page_count = source.page_count

    paper.meta = {
        **(paper.meta or {}),
        "sections": len(source.sections),
        "source_type": source.source_type,
        # 引用键留着：后续建引用关系图要用（LaTeX 才有，PDF 拿不到）
        "citations": source.citations[:500],
        "notes": source.notes[:5],
    }
    db.session.commit()

    log.info(
        "索引完成 %s：来源 %s，%d 节 -> %d 块",
        paper.id, source.source_type, len(source.sections), len(drafts),
    )
    return len(drafts)


def _enrich_from_source(paper: Paper, source) -> None:
    """用解析结果补全论文元数据。

    只在字段为空时填——用户手工改过的标题不该被重新解析覆盖掉。
    """
    from .pdf import normalize_title

    if source.title and (not paper.title or (paper.meta or {}).get("title_source") in {"filename", None}):
        paper.title = source.title
        paper.title_norm = normalize_title(source.title)
        paper.meta = {**(paper.meta or {}), "title_source": f"{source.source_type}_parse"}

    if source.authors and not paper.authors:
        paper.authors = source.authors
    if source.abstract and not paper.abstract:
        paper.abstract = source.abstract


def _enrich_metadata(paper: Paper, document) -> None:
    """用全文里能拿到的信息补全元数据。

    扫描阶段为了速度只读首页，有些标识符（尤其是正文里引用的 DOI）
    这里才能补上。
    """
    from .pdf import find_arxiv_id, find_doi, normalize_arxiv, normalize_doi

    if not paper.abstract and len(document.pages) > 1:
        from .pdf import _extract_abstract

        head = "\n".join(p.text for p in document.pages[:2])
        abstract = _extract_abstract(head)
        if abstract:
            paper.abstract = abstract

    if not paper.doi or not paper.arxiv_id:
        head = "\n".join(p.text for p in document.pages[:2])
        if not paper.doi:
            paper.doi = normalize_doi(find_doi(head))
        if not paper.arxiv_id:
            paper.arxiv_id = normalize_arxiv(find_arxiv_id(head))


def reindex_all(ctx=None) -> dict:
    """全量重建索引：清空所有分块后重新来过。

    换分块参数、或修复索引损坏时用。比逐篇 force 更彻底，也更慢。
    """
    from .fts import rebuild

    if ctx is not None:
        ctx.progress(0.05, "正在清空旧索引…")

    count = db.session.query(Chunk).filter(Chunk.note_id.is_(None)).delete(synchronize_session=False)
    db.session.query(Paper).filter(Paper.deleted_at.is_(None)).update(
        {"ingest_status": INGEST_PENDING}, synchronize_session=False
    )
    db.session.commit()
    log.info("已清空 %d 个分块，准备重建", count)

    result = index_papers(ctx=ctx, force=False)
    result["cleared"] = count

    if ctx is not None:
        ctx.progress(0.99, "重建全文索引…")
    rebuild(db.engine)
    return result


def index_stats() -> dict:
    """索引状态概览。"""
    from sqlalchemy import func

    total_chunks = db.session.query(Chunk).count()
    paper_chunks = db.session.query(Chunk).filter(Chunk.note_id.is_(None)).count()
    note_chunks = db.session.query(Chunk).filter(Chunk.note_id.isnot(None)).count()

    indexed_papers = (
        db.session.query(Paper).filter(Paper.ingest_status == INGEST_INDEXED).count()
    )
    failed_papers = db.session.query(Paper).filter(Paper.ingest_status == INGEST_FAILED).count()
    pending_papers = (
        db.session.query(Paper)
        .filter(Paper.deleted_at.is_(None), Paper.ingest_status == INGEST_PENDING)
        .count()
    )

    last_indexed = db.session.query(func.max(Paper.updated_at)).filter(
        Paper.ingest_status == INGEST_INDEXED
    ).scalar()

    return {
        "chunks_total": total_chunks,
        "chunks_paper": paper_chunks,
        "chunks_note": note_chunks,
        "papers_indexed": indexed_papers,
        "papers_pending": pending_papers,
        "papers_failed": failed_papers,
        "last_indexed_at": last_indexed,
    }


def index_note(note) -> int:
    """把一篇笔记的正文切成块写进索引。返回块数。

    **先删后插**，和论文索引同一套策略：笔记的改动可能增删小节，
    增量更新对不上号。笔记都很短，整篇重来的代价可以忽略。

    注意 ``kind`` 用的是 ``CHUNK_NOTE`` 而不是 ``CHUNK_TEXT``——
    检索时「论文原文」和「我自己的笔记」要能分开：前者是别人的结论，
    后者是我读过之后的判断，把两者混为一谈会让人分不清哪句话是谁说的。
    """
    from .chunker import chunk_markdown, rules_meta

    db.session.query(Chunk).filter(Chunk.note_id == note.id).delete(
        synchronize_session=False
    )

    sections = chunk_markdown(note.content_md or "")
    if not sections:
        db.session.commit()
        return 0

    for order, section in enumerate(sections):
        db.session.add(
            Chunk(
                note_id=note.id,
                # 论文也挂在笔记上，检索结果要能顺着笔记找到它对应的论文
                paper_id=note.paper_id,
                kind=CHUNK_NOTE,
                # 引用定位符就是小节标题，形如「方法」——
                # 笔记没有页码，用户要的是「哪一节」而不是「第几页」
                section_path=section.heading,
                ord=order,
                text=section.text,
                n_tokens=section.n_tokens,
                is_section_start=True,
                meta={
                    "section_title": section.heading,
                    "source": "note",
                    **rules_meta(),
                },
            )
        )
    db.session.commit()
    return len(sections)


def reindex_note_safely(note) -> None:
    """重建笔记索引，**失败只记日志**。

    笔记索引挂在保存路径上（网页保存、外部编辑同步、流水线发布）。
    索引失败不能把这些操作搞失败：用户按下保存，要的是内容存下来；
    「能不能搜到」是次要的，而且可以事后用 ``kb index-notes`` 补建。
    反过来让索引异常把保存打断，是把主次搞反了。
    """
    try:
        index_note(note)
    except Exception:
        log.warning("笔记 %s 的索引重建失败（内容已保存）", note.id, exc_info=True)


def index_notes(notes=None) -> dict:
    """批量重建笔记索引。``notes`` 为空时处理全部笔记。"""
    from ..models import Note

    query = db.session.query(Note)
    if notes is not None:
        query = query.filter(Note.id.in_([n.id for n in notes]))
    rows = query.all()

    total_chunks = 0
    for note in rows:
        total_chunks += index_note(note)
    log.info("笔记索引完成：%d 篇 -> %d 块", len(rows), total_chunks)
    return {"notes": len(rows), "chunks": total_chunks}


def note_index_report() -> dict:
    """笔记索引的健康度：有多少篇笔记根本没进索引。

    **这个数字必须能看见。** 笔记索引加进来之前，全库 85 篇笔记一块都没索引，
    而任何地方都不报错——检索少了一大块内容，表现只是「有些东西搜不到」。
    靠人记得去查是靠不住的，要让它在自检里自己冒出来。
    """
    from ..models import Note

    total_notes = db.session.query(Note).count()
    indexed_notes = (
        db.session.query(db.func.count(db.distinct(Chunk.note_id)))
        .filter(Chunk.note_id.isnot(None))
        .scalar()
        or 0
    )
    note_chunks = db.session.query(Chunk).filter(Chunk.note_id.isnot(None)).count()
    return {
        "notes": total_notes,
        "indexed_notes": indexed_notes,
        "missing": max(0, total_notes - indexed_notes),
        "chunks": note_chunks,
        "up_to_date": total_notes == indexed_notes,
    }


def stale_chunk_report() -> dict:
    """哪些块是用旧规则建的。

    **这条信息本身不修任何东西**，它只是把「检索质量悄悄变差」变成
    一个能看见的数字。分块规则改了之后不重建索引，库里的块会新旧混在一起：
    检索仍然能跑、仍然返回结果、也不报错——只是结果比应有的差。

    这正是最难发现的一类退化：没有任何一处会失败。
    """
    from collections import Counter

    rows = (
        db.session.query(Chunk.meta, Chunk.paper_id)
        .filter(Chunk.note_id.is_(None))
        .all()
    )
    by_version: Counter = Counter()
    stale_papers: set[str] = set()
    for meta, paper_id in rows:
        # 没有 rules 字段的块是「引入版本号之前建的」。不猜它是几版——
        # 旧版本的真实来源无法还原，当成需要重建的最稳妥。
        version = (meta or {}).get("rules")
        by_version[version] += 1
        if version != RULES_VERSION and paper_id:
            stale_papers.add(paper_id)

    current = by_version.get(RULES_VERSION, 0)
    return {
        "rules_version": RULES_VERSION,
        "current": current,
        "by_version": {
            # None 在 JSON 里会变成 null，前端不好显示，转成可读的名字
            ("未标记" if key is None else str(key)): count
            for key, count in sorted(
                by_version.items(), key=lambda kv: (kv[0] is not None, kv[0])
            )
        },
        "stale": len(rows) - current,
        "stale_papers": len(stale_papers),
        "up_to_date": current == len(rows),
    }


def drop_paper_chunks(paper_id: str) -> int:
    """删除某篇论文的全部分块（连带清理 FTS 索引）。"""
    count = (
        db.session.query(Chunk)
        .filter(Chunk.paper_id == paper_id, Chunk.note_id.is_(None))
        .delete(synchronize_session=False)
    )
    db.session.commit()
    return count


__all__ = [
    "drop_paper_chunks",
    "index_note",
    "index_notes",
    "index_paper",
    "index_papers",
    "index_stats",
    "note_index_report",
    "reindex_all",
    "reindex_note_safely",
    "stale_chunk_report",
]
