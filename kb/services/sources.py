r"""正文来源的获取与归一。

论文正文有两个来源，各有优劣：

                LaTeX 源码                        PDF
  结构      \section 显式声明，层级准确        靠字号/缩进猜，双栏易错
  公式      原始 LaTeX，可直接渲染            抽取成文本后大量符号丢失
  引用键    \cite{key} 直接可得               只有 "[1]"，要反解参考文献表
  图注      与图一一对应                      位置关系需要推测
  页码      没有（要编译才知道）              天然就有

**所以优先用源码，但用 PDF 补上页码。** 具体做法是：解析出 LaTeX 的章节后，
拿每个章节标题去 PDF 里搜索，找到它出现在第几页。这样既拿到了源码的准确结构，
又保住了引用所需的页码定位。

拿不到源码（作者只传了 PDF、非 arXiv 论文、网络不可用）时，完全退回到 PDF 解析。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Paper

log = logging.getLogger(__name__)

SOURCE_LATEX = "latex"
SOURCE_PDF = "pdf"


@dataclass
class SourceResult:
    """归一化之后的正文来源。"""

    source_type: str
    sections: list  # list[chunker.Section]
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    page_count: int = 0
    # 公式与图注的独立分块。结构：{text, kind, section_path, page, label}
    extra_chunks: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    bibitems: dict[str, str] = field(default_factory=dict)
    equations: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    source_dir: Path | None = None
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.sections)


# --------------------------------------------------------------------------
# 本地源码
# --------------------------------------------------------------------------


def find_local_source(pdf_path: str | Path) -> Path | None:
    """在 PDF 旁边找现成的 LaTeX 源码。

    很多人会把 arXiv 的源码包和解压后的目录跟 PDF 放在一起。
    用现成的比重新下载快，也不消耗 arXiv 的配额。
    """
    pdf = Path(pdf_path)
    parent = pdf.parent
    stem = pdf.stem

    # 同名目录
    for candidate in (parent / stem, parent / f"{stem}-src", parent / f"{stem}_src"):
        if candidate.is_dir() and any(candidate.rglob("*.tex")):
            return candidate

    # 同名的压缩包
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        archive = parent / f"{stem}{suffix}"
        if archive.is_file():
            return archive

    # 目录本身就是源码目录（比如论文目录里直接放着一堆 .tex）
    tex_files = list(parent.glob("*.tex"))
    if len(tex_files) >= 2:
        return parent

    return None


def unpack_local_archive(archive: Path, dest: Path) -> tuple[Path | None, str]:
    """解开本地的源码压缩包。

    不用 ``extractall``：它会把成员名里的 ``..`` 或绝对路径原样用于写入，
    从而把文件落到目标目录之外。压缩包是外部输入（从 arXiv 下载或用户自备），
    必须逐个成员校验。Python 3.12+ 的 ``filter="data"`` 能做这件事，
    但为了兼容更早的解释器，这里自己实现一遍。
    """
    import tarfile
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()

    def safe_target(name: str) -> Path | None:
        target = (dest_resolved / name).resolve()
        if target != dest_resolved and not str(target).startswith(str(dest_resolved) + os.sep):
            log.warning("跳过越界的归档成员：%s", name)
            return None
        return target

    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    if info.is_dir():
                        continue
                    target = safe_target(info.filename)
                    if target is None:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, open(target, "wb") as sink:
                        shutil.copyfileobj(source, sink)
            return dest, ""

        if tarfile.is_tarfile(archive):
            from .latex import _safe_extract

            with tarfile.open(archive) as bundle:
                _safe_extract(bundle, dest)
            return dest, ""

    except Exception as exc:
        return None, f"解包失败：{exc}"

    return None, "不是可识别的压缩包"


# --------------------------------------------------------------------------
# 从 LaTeX 构建章节
# --------------------------------------------------------------------------


def _latex_sections_to_chunker(
    latex_document,
    *,
    page_lookup: dict[str, int] | None = None,
) -> list:
    """把 LaTeX 章节转成 chunker 认识的 ``Section``。

    关键的一步是**用 PDF 反查页码**：LaTeX 本身不含页码，但没有页码的引用
    （「见 §3.2」）在阅读器里没法跳转。做法是拿章节标题去 PDF 里搜——
    标题是作者自己写的、通常唯一，搜到的第一处就是该节起始页。
    """
    from .chunker import Section
    from .latex import extract_section_text

    sections: list[Section] = []
    extras: list[dict] = []

    for latex_section in latex_document.sections:
        # 正文在 _walk_sections 切分时就已经就地取到（section.raw_body），
        # 这里只做转换，不再有「第二遍对齐」的问题
        raw = latex_section.raw_body or ""
        paragraphs = extract_section_text(raw) if raw else list(latex_section.paragraphs)

        # 公式与图注**不**混进散文段落，而是作为独立分块单独返回。
        #
        # 理由：它们的检索语义与散文不同。「注意力的公式是什么」该直接命中
        # 公式本身，而不是某个恰好包含它的长段落；「表 1 报的复杂度是多少」
        # 同理。混在一起还会让分块器把公式合并进相邻段落，
        # 结果公式那几行被淹没在几百字里，检索时权重被稀释。
        if not paragraphs:
            continue

        page = 0
        if page_lookup:
            page = page_lookup.get(latex_section.title, 0)
            if not page and latex_section.label:
                page = page_lookup.get(latex_section.label, 0)

        sections.append(
            Section(
                title=latex_section.title,
                level=latex_section.level,
                path=latex_section.path,
                page_from=page or 1,
                page_to=page or 1,
                paragraphs=paragraphs,
                paragraph_pages=[page or 1] * len(paragraphs),
            )
        )

        # 公式：保留原始 LaTeX，前端的 KaTeX 会渲染成真正的数学排版。
        # 标题里带上它所属的小节，检索结果才有上下文。
        for equation in latex_section.equations:
            label = f"（{equation.label}）" if equation.label else ""
            # 有编号的公式更可能被引用，排在前面
            prefix = "公式" if equation.numbered else "行间公式"
            extras.append(
                {
                    "text": f"{prefix}{label}：{equation.latex}",
                    "kind": "formula",
                    "section_path": latex_section.path,
                    "page": page or 1,
                    "meta": {"latex": equation.latex, "eq_label": equation.label},
                }
            )

        for figure in latex_section.figures:
            if not figure.caption and not figure.body:
                continue
            kind_label = "图" if figure.kind == "figure" else "表"
            # 表格正文要一并写进 text，不能只留标题。
            # 此前只发「表注：…」，于是整个索引里没有一个表格数值——
            # 笔记里 79% 的 ⚠️ 待核都在说「具体数值未在提供的文本中给出」，
            # 而表格就在源码里躺着。见 latex.render_tabular 的说明。
            body = (figure.body or "").strip()
            if body:
                text = f"{kind_label}注：{figure.caption}\n\n{body}" if figure.caption else body
            else:
                text = f"{kind_label}注：{figure.caption}"
            extras.append(
                {
                    "text": text,
                    "kind": "figure" if figure.kind == "figure" else "table",
                    "section_path": latex_section.path,
                    "page": page or 1,
                    "meta": {
                        "caption": figure.caption,
                        "graphics": figure.graphics,
                        "fig_label": figure.label,
                        "has_body": bool(body),
                    },
                }
            )

    return sections, extras


def build_page_lookup(sections, pdf_path: str | Path | None) -> dict[str, int]:
    """拿章节标题去 PDF 里搜，得到「标题 -> 页码」。

    搜不到就返回 0，调用方会退化成「只有章节号、没有页码」的引用——
    这仍然比没有引用强。
    """
    if not pdf_path or not Path(pdf_path).is_file():
        return {}

    try:
        import pymupdf
    except ImportError:
        return {}

    lookup: dict[str, int] = {}
    try:
        document = pymupdf.open(str(pdf_path))
    except Exception:
        return {}

    try:
        for section in sections:
            title = (section.title or "").strip()
            if len(title) < 4:
                continue
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                # 用文本搜索而不是 get_text 后自己找：search_for 会处理
                # 连字、多空格这类排版差异，命中的概率更高
                if page.search_for(title):
                    lookup[title] = page_index + 1
                    break
    except Exception:
        log.debug("用 PDF 反查章节页码时出错", exc_info=True)
    finally:
        document.close()

    if lookup:
        log.debug("PDF 反查得到 %d/%d 个章节的页码", len(lookup), len(sections))
    return lookup


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def acquire_source(paper: Paper) -> SourceResult:
    """取得一篇论文的正文来源。

    顺序：本地源码 -> arXiv 下载 -> PDF 解析。
    """
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    cfg = current_app.extensions["kb_boot_config"]

    prefer_latex = bool(settings.get("ingest.prefer_latex"))
    allow_fetch = bool(settings.get("ingest.fetch_arxiv_source"))
    pdf_path = paper.file_path

    if prefer_latex:
        result = _try_latex(paper, cfg, allow_fetch=allow_fetch, pdf_path=pdf_path)
        if result is not None and result.ok:
            result.page_count = _pdf_page_count(pdf_path)
            return result
        if result is not None and result.errors:
            log.info("LaTeX 路径未成功（%s），退回 PDF", "; ".join(result.errors[:2]))

    return _from_pdf(paper, settings)


def _try_latex(paper: Paper, cfg, *, allow_fetch: bool, pdf_path: str | None):
    """尝试走 LaTeX 路径。返回 None 表示连尝试都没做成。"""
    from .latex import (
        LatexError,
        fetch_arxiv_source,
        find_main_tex,
        parse_latex,
    )

    source_dir: Path | None = None
    errors: list[str] = []

    # 1) 本地已有的源码
    if pdf_path:
        local = find_local_source(pdf_path)
        if local is not None:
            if local.is_file():
                unpacked, error = unpack_local_archive(
                    local, cfg.cache_dir / "sources" / paper.id
                )
                if unpacked is not None:
                    source_dir = unpacked
                else:
                    errors.append(error)
            else:
                source_dir = local

    # 2) 从 arXiv 下载
    if source_dir is None and allow_fetch and paper.arxiv_id:
        target = cfg.cache_dir / "sources" / (paper.arxiv_id.replace("/", "_"))
        # 已经下载过就直接复用，避免重复消耗 arXiv 的配额
        if target.is_dir() and any(target.rglob("*.tex")):
            source_dir = target
        else:
            try:
                source_dir, error = fetch_arxiv_source(paper.arxiv_id, target)
                if source_dir is None:
                    errors.append(error or "下载失败")
            except LatexError as exc:
                errors.append(str(exc))

    if source_dir is None:
        return _empty_result(errors or ["没有可用的 LaTeX 源码"])

    main = find_main_tex(source_dir)
    if main is None:
        return _empty_result([*errors, "源码包里没有找到主 .tex 文件"])

    try:
        text = main.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _empty_result([*errors, f"读取主文档失败：{exc}"])

    try:
        document = parse_latex(text, source_dir=source_dir)
    except Exception as exc:
        log.exception("解析 LaTeX 失败")
        return _empty_result([*errors, f"解析失败：{exc}"])

    if not document.sections:
        return _empty_result([*errors, "解析后没有得到任何章节"])

    page_lookup = build_page_lookup(document.sections, pdf_path)
    sections, extras = _latex_sections_to_chunker(document, page_lookup=page_lookup)

    if not sections:
        return _empty_result([*errors, "章节里没有正文内容"])

    result = SourceResult(
        source_type=SOURCE_LATEX,
        sections=sections,
        extra_chunks=extras,
        title=document.title,
        authors=document.authors,
        abstract=document.abstract,
        citations=document.citations,
        bibitems=document.bibitems,
        equations=[e for s in document.sections for e in s.equations],
        figures=[f for s in document.sections for f in s.figures],
        source_dir=source_dir,
        errors=[*errors, *document.errors],
    )
    covered = len(page_lookup)
    result.notes.append(
        f"正文来自 LaTeX 源码（{len(sections)} 节，{len(result.citations)} 条引用），"
        + (f"{covered} 个章节已反查到 PDF 页码。" if covered else "未能在 PDF 中定位页码。")
    )
    return result




def _from_pdf(paper: Paper, settings) -> SourceResult:
    """PDF 路径。

    优先用 ``pymupdf4llm``：它把 PDF 转成 Markdown，标题层级识别得比按字号
    猜准得多。实测 Attention Is All You Need，它能正确还原
    「3.2.1 Scaled Dot-Product Attention」这样的带编号层级，
    而字号启发式只能给出扁平的几节，子章节全部丢失。

    它不可用时退回 PyMuPDF + 启发式——功能不丢，只是结构粗一些。
    """
    from .chunker import detect_sections
    from .pdf import PdfError, extract_document

    if not paper.file_path:
        return _empty_result(["论文没有关联文件"])

    engine = (settings.get("ingest.pdf_engine") or "auto").lower()

    if engine in {"auto", "markdown"}:
        result = _from_pdf_markdown(paper)
        if result is not None and result.ok:
            return result
        if engine == "markdown":
            return result or _empty_result(["Markdown 转换未产出内容"])

    try:
        document = extract_document(paper.file_path)
    except PdfError as exc:
        return _empty_result([str(exc)])

    sections = detect_sections(document.pages, document.outline)

    return SourceResult(
        source_type=SOURCE_PDF,
        sections=sections,
        title=document.metadata.title,
        authors=document.metadata.authors,
        abstract=document.metadata.abstract or "",
        page_count=len(document.pages),
        notes=["正文来自 PDF（按字号启发式识别章节），没有可用的 LaTeX 源码。"],
    )


# Markdown 标题行：`## **3.2 Attention**`、`### 3.2.1 Scaled Dot-Product`
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# 标题常带粗体/斜体标记，要剥掉
_MD_TITLE_CLEAN = re.compile(r"^\*\*(.*?)\*\*$|^\*(.*?)\*$")


def _clean_md_title(raw: str) -> str:
    text = raw.strip()
    match = _MD_TITLE_CLEAN.match(text)
    if match:
        text = (match.group(1) or match.group(2) or "").strip()
    return text.strip("#* ").strip()


def _from_pdf_markdown(paper: Paper) -> SourceResult | None:
    """用 pymupdf4llm 把 PDF 转成 Markdown 再解析结构。

    用 ``page_chunks=True`` 拿分页结果，这样每个段落都知道自己在第几页——
    引用能精确到页是 PDF 路径相对 LaTeX 路径的唯一优势，不能丢。
    """
    try:
        import pymupdf4llm
    except ImportError:
        return None

    if not paper.file_path:
        return None

    try:
        pages = pymupdf4llm.to_markdown(
            str(paper.file_path), page_chunks=True, show_progress=False
        )
    except Exception as exc:
        log.warning("pymupdf4llm 转换失败，退回启发式解析：%s", exc)
        return None

    if not pages:
        return None

    from .chunker import Section

    sections: list[Section] = []
    parents: dict[int, str] = {}
    doc_title = ""
    state = {"current": None, "seen_heading": False}

    def flush() -> None:
        current = state["current"]
        if current is not None and current.paragraphs:
            sections.append(current)
        state["current"] = None

    for page in pages:
        page_no = int((page.get("metadata") or {}).get("page_number") or 0) or 1
        text = page.get("text") or ""
        buffer: list[str] = []

        for line in text.split("\n"):
            heading = _MD_HEADING.match(line)
            if heading is None:
                buffer.append(line)
                continue

            if buffer:
                if state["current"] is not None:
                    _push_paragraph(state["current"], "\n".join(buffer), page_no)
                buffer = []

            flush()
            level = len(heading.group(1))
            title = _clean_md_title(heading.group(2))
            if not title:
                continue

            # 文档开头的一级标题是论文标题，不是章节。
            # 不跳过去的话，所有章节路径都会带上它（「Attention Is All You
            # Need > 5 Training > 5.3 Optimizer」），每个引用都拖一串冗余前缀。
            if not state["seen_heading"] and level == 1:
                state["seen_heading"] = True
                if not doc_title:
                    doc_title = title
                continue
            state["seen_heading"] = True

            parents[level] = title
            for deeper in [k for k in parents if k > level]:
                del parents[deeper]

            state["current"] = Section(
                title=title,
                level=level,
                path=" > ".join(parents[k] for k in sorted(parents)),
                page_from=page_no,
                page_to=page_no,
            )

        if buffer:
            if state["current"] is None:
                # 第一个标题出现之前的正文（首页的标题、作者、摘要开头）
                state["current"] = Section(
                    title="", level=0, path="", page_from=page_no, page_to=page_no
                )
            _push_paragraph(state["current"], "\n".join(buffer), page_no)

    flush()

    if not sections:
        return None

    meta = pages[0].get("metadata") or {}
    return SourceResult(
        source_type=SOURCE_PDF,
        sections=sections,
        # 优先用 PDF 元数据里的标题；没有就用正文里那个一级标题
        title=(meta.get("title") or "").strip() or doc_title,
        page_count=int(meta.get("page_count") or len(pages)),
        notes=["正文来自 PDF（pymupdf4llm 转 Markdown 后解析），没有可用的 LaTeX 源码。"],
    )


def _push_paragraph(section, block: str, page_no: int) -> None:
    """把一段 Markdown 文本按空行拆成段落塞进章节。

    表格整块保留、不按行拆：表格按行拆开后就完全没法读了，
    而表格往往是「这个指标是多少」这类问题的唯一答案所在。
    """
    for piece in re.split(r"\n\s*\n", block):
        text = piece.strip()
        if len(text) < 20:
            continue
        section.paragraphs.append(text)
        section.paragraph_pages.append(page_no)
        section.page_to = page_no


def _empty_result(errors: list[str]) -> SourceResult:
    return SourceResult(source_type=SOURCE_PDF, sections=[], errors=list(errors))


def _pdf_page_count(pdf_path: str | None) -> int:
    if not pdf_path or not Path(pdf_path).is_file():
        return 0
    try:
        import pymupdf

        with pymupdf.open(str(pdf_path)) as document:
            return document.page_count
    except Exception:
        return 0


def clear_source_cache(cfg, paper_id: str | None = None) -> int:
    """清掉下载缓存。"""
    root = cfg.cache_dir / "sources"
    if not root.is_dir():
        return 0
    target = root / paper_id if paper_id else root
    if not target.exists():
        return 0
    shutil.rmtree(target, ignore_errors=True)
    return 1


__all__ = [
    "SOURCE_LATEX",
    "SOURCE_PDF",
    "SourceResult",
    "acquire_source",
    "build_page_lookup",
    "find_local_source",
]
