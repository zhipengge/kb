"""PDF 读取。

分成两个层次，因为它们的使用场景对性能的要求完全不同：

  * **轻量读取**（本模块上半部分）：只为扫描目录时拿到标题、作者、DOI/arXiv 号。
    扫描可能要处理上万个文件，每个文件都全文解析是不可接受的——这里只读
    元数据字典和首页文本。
  * **完整解析**（``extract_document``）：抽全文、分页、识别章节结构，
    只在真正要对一篇论文建索引或深度阅读时执行。

许可提示：本项目默认使用 PyMuPDF（``import pymupdf``），其许可为 **AGPL-3.0**。
个人自用无影响；闭源分发需要替换实现。所有 PyMuPDF 调用都收敛在本模块内，
替换成 pdfplumber/pypdf 只需要改这一个文件。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# arXiv 编号：1706.03762 或 1706.03762v5；也有 cs/0701001 这样的老格式
_ARXIV_RE = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7})", re.I)
_ARXIV_BARE_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")

# DOI：10.xxxx/后面跟任意允许字符
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.I)

# 常见的「不是标题」的首页行
_TITLE_NOISE = re.compile(
    r"^(arxiv:|preprint|published as|to appear|under review|proceedings of|"
    r"ieee|cvpr|icml|iclr|neurips|nips|acl|emnlp|aaai|ijcai|www\.|http)",
    re.I,
)


@dataclass
class PdfMetadata:
    """从 PDF 里能可靠拿到的信息。"""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    page_count: int = 0
    title_source: str = ""  # metadata / first_page / filename


class PdfError(RuntimeError):
    """PDF 打不开或不是有效的 PDF。"""


def _open(path: str | Path):
    import pymupdf  # 延迟导入：CLI 的纯配置命令不该为它付出启动开销

    try:
        return pymupdf.open(str(path))
    except Exception as exc:
        raise PdfError(f"无法打开 PDF：{exc}") from exc


# --------------------------------------------------------------------------
# 识别符提取
# --------------------------------------------------------------------------


def find_arxiv_id(text: str) -> str | None:
    match = _ARXIV_RE.search(text)
    if match:
        return match.group(1)
    # 首页里也可能只写裸编号（很多论文模板的边栏就是这样）
    match = _ARXIV_BARE_RE.search(text)
    if match:
        candidate = match.group(1)
        # 裸编号容易误伤（年份、页码、公式编号），要求它出现在明显的位置
        year = int(candidate[:2])
        if 7 <= year <= 99:
            return candidate + (match.group(2) or "")
    return None


def find_doi(text: str) -> str | None:
    match = _DOI_RE.search(text)
    if not match:
        return None
    doi = match.group(1).rstrip(".,;)")
    # 去掉尾部常见的粘连标点
    for suffix in ("</", "&gt", "&lt"):
        doi = doi.split(suffix)[0]
    return doi or None


def normalize_doi(doi: str | None) -> str | None:
    """DOI 大小写不敏感，统一小写便于比较。"""
    if not doi:
        return None
    return doi.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")


def normalize_arxiv(arxiv_id: str | None) -> str | None:
    """去掉版本号后缀——同一篇论文的 v1 和 v3 是同一篇。"""
    if not arxiv_id:
        return None
    return re.sub(r"v\d+$", "", arxiv_id.strip(), flags=re.I)


# --------------------------------------------------------------------------
# 标题
# --------------------------------------------------------------------------


def normalize_title(title: str | None) -> str:
    """标题归一化，用于模糊比对。

    去掉大小写、标点、多余空白，以及 LaTeX 里常见的花括号。
    目的是让 "Attention Is All You Need"、"attention is all you need."
    和 "{Attention} Is All You Need" 归一到同一个键。
    """
    if not title:
        return ""
    text = str(title)
    text = re.sub(r"[{}]", "", text)          # LaTeX 保护花括号
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)  # 标点变空格（保留 CJK）
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def title_from_filename(path: str | Path) -> str:
    """从文件名猜标题。

    去扩展名，把下划线和连字符还原成空格，去掉常见的下载后缀
    （(1)、-copy、arXiv 编号前缀等）。
    """
    stem = Path(path).stem
    stem = re.sub(r"^\d{4}\.\d{4,5}v?\d*[_\-\s]*", "", stem)     # 开头的 arXiv 号
    stem = re.sub(r"\((?:19|20)\d{2}\)", "", stem)                # (2023)
    stem = re.sub(r"[_\s]+", " ", stem)
    stem = re.sub(r"\s*-\s*", " ", stem)
    stem = re.sub(r"\s*(copy|final|preprint|draft)\s*\d*$", "", stem, flags=re.I)
    stem = re.sub(r"\s+", " ", stem).strip(" .-_")
    return stem or Path(path).stem


def _looks_like_title(line: str) -> bool:
    text = line.strip()
    if len(text) < 12 or len(text) > 300:
        return False
    if _TITLE_NOISE.match(text):
        return False
    # 标题一般不是以句号结尾的完整句子
    if text.endswith(".") and len(text.split()) > 12:
        return False
    # 邮箱、URL 之类
    if "@" in text and "." in text.split("@")[-1][:6]:
        return False
    return True


def _title_from_first_page(page) -> str:
    """从首页挑出最可能是标题的一行。

    策略：取字号最大的文本块，用「字号 × 靠上程度」排序。这是启发式，
    不会 100% 准确——所以调用方要保留来源标记，界面上允许用户手工纠正。
    """
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return ""

    candidates: list[tuple[float, str]] = []
    page_height = page.rect.height or 1.0

    for block in blocks:
        if block.get("type") != 0:  # 0=文本，1=图片
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not _looks_like_title(text):
                continue
            size = max((s.get("size", 0) for s in spans), default=0)
            y = line.get("bbox", (0, page_height * 2))[1]
            # 越靠上、字号越大越可能是标题
            y_ratio = 1.0 - min(y / page_height, 1.0)
            score = size * (0.5 + y_ratio)
            candidates.append((score, text))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _extract_abstract(text: str) -> str | None:
    """从正文里截出摘要段。"""
    match = re.search(
        r"\babstract\b[\s:—-]*(.{80,2500}?)(?:\n\s*\n|\b(?:1|I)\.?\s+(?:introduction|引言))",
        text,
        re.I | re.S,
    )
    if not match:
        return None
    abstract = re.sub(r"\s+", " ", match.group(1)).strip()
    return abstract[:2000] or None


# --------------------------------------------------------------------------
# 轻量读取（扫描时用）
# --------------------------------------------------------------------------


def read_metadata(path: str | Path, *, read_first_page: bool = True) -> PdfMetadata:
    """读取标题、作者、标识符等。

    ``read_first_page=False`` 时只读 PDF 自带的元数据字典，速度极快，
    适合「文件内容没变、只想确认一下」的场景。
    """
    doc = _open(path)
    try:
        meta = PdfMetadata(page_count=doc.page_count)

        info = doc.metadata or {}
        raw_title = (info.get("title") or "").strip()
        raw_author = (info.get("author") or "").strip()

        # PDF 元数据里的标题常常是 LaTeX 模板留下的垃圾（"untitled"、
        # 文件名、\LaTeX 的默认值），要能识别出来
        if raw_title and len(raw_title) > 6 and not _is_junk_title(raw_title, path):
            meta.title = raw_title
            meta.title_source = "metadata"

        if raw_author:
            meta.authors = _split_authors(raw_author)

        first_page_text = ""
        if read_first_page and doc.page_count:
            page = doc.load_page(0)
            first_page_text = page.get_text("text") or ""

            if not meta.title:
                guessed = _title_from_first_page(page)
                if guessed:
                    meta.title = guessed
                    meta.title_source = "first_page"

        if not meta.title:
            meta.title = title_from_filename(path)
            meta.title_source = "filename"

        # 标识符可能出现在首页任意位置，也可能在元数据的 subject/keywords 里
        haystack = first_page_text
        if not haystack:
            for key in ("subject", "keywords", "title"):
                value = info.get(key)
                if value:
                    haystack += f" {value}"

        if haystack:
            meta.arxiv_id = normalize_arxiv(find_arxiv_id(haystack))
            meta.doi = normalize_doi(find_doi(haystack))
            meta.abstract = _extract_abstract(first_page_text)

        # 年份：优先元数据的 creationDate，退而求其次用 arXiv 号前缀
        meta.year = _year_from_info(info) or _year_from_arxiv(meta.arxiv_id)

        return meta
    finally:
        doc.close()


def _is_junk_title(title: str, path: str | Path) -> bool:
    lowered = title.strip().lower()
    if lowered in {"untitled", "microsoft word", "document", "paper", "article", "manuscript"}:
        return True
    # 标题就是文件名（Word 导出 PDF 的常见行为）
    return lowered == Path(path).stem.lower()


def _split_authors(raw: str) -> list[str]:
    """拆分作者串。

    分隔符在真实数据里五花八门：逗号、分号、and、&、多个空格。
    """
    if not raw:
        return []
    parts = re.split(r"\s*(?:,|;|\band\b|&|·)\s*", raw)
    authors = [p.strip() for p in parts if p.strip()]
    # 过滤掉明显的非人名（超长的字符串通常是机构名）
    return [a for a in authors if 1 < len(a) < 80][:50]


def _year_from_info(info: dict) -> int | None:
    for key in ("creationDate", "modDate"):
        value = str(info.get(key) or "")
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            year = int(match.group(0))
            if 1900 <= year <= 2100:
                return year
    return None


def _year_from_arxiv(arxiv_id: str | None) -> int | None:
    if not arxiv_id:
        return None
    match = re.match(r"(\d{2})", arxiv_id)
    if not match:
        return None
    year = int(match.group(1))
    # arXiv 从 1991 年开始，两位年份 > 91 属于上个世纪
    return 1900 + year if year > 91 else 2000 + year


# --------------------------------------------------------------------------
# 完整解析（建索引 / 深度阅读时用）
# --------------------------------------------------------------------------


@dataclass
class PdfPage:
    number: int  # 1-based，与 PDF 阅读器显示一致
    text: str
    width: float = 0.0
    height: float = 0.0


@dataclass
class PdfDocument:
    metadata: PdfMetadata
    pages: list[PdfPage]
    outline: list[tuple[int, str, int]] = field(default_factory=list)  # (层级, 标题, 页码)


def extract_document(path: str | Path, *, max_pages: int | None = None) -> PdfDocument:
    """完整抽取：逐页文本 + 目录结构。

    分页保留是刻意的——每一页的文本都单独存，分块时才能知道每块来自第几页，
    引用才能精确到页。
    """
    doc = _open(path)
    try:
        meta = read_metadata(path)
        pages: list[PdfPage] = []

        limit = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for index in range(limit):
            page = doc.load_page(index)
            rect = page.rect
            pages.append(
                PdfPage(
                    number=index + 1,
                    text=page.get_text("text") or "",
                    width=rect.width,
                    height=rect.height,
                )
            )

        outline: list[tuple[int, str, int]] = []
        try:
            for level, title, page_number in doc.get_toc() or []:
                outline.append((int(level), str(title).strip(), int(page_number)))
        except Exception:
            log.debug("读取 PDF 目录结构失败，将使用启发式章节识别")

        return PdfDocument(metadata=meta, pages=pages, outline=outline)
    finally:
        doc.close()


def looks_like_pdf(path: str | Path) -> bool:
    """检查文件头魔术字节。

    不能只看扩展名：上传接口收到的是外部输入，把 .exe 改名成 .pdf 是最基本的
    攻击手法。这里只读前 5 个字节，成本可以忽略。
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


__all__ = [
    "PdfDocument",
    "PdfError",
    "PdfMetadata",
    "PdfPage",
    "extract_document",
    "find_arxiv_id",
    "find_doi",
    "looks_like_pdf",
    "normalize_arxiv",
    "normalize_doi",
    "normalize_title",
    "read_metadata",
    "title_from_filename",
]
