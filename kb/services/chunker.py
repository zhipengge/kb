"""章节识别与文本分块。

分块质量直接决定检索质量，而**引用能不能精确定位**取决于这里记了什么。
所以每个块都带上 ``page_from`` / ``page_to`` / ``section_path``——
「论文 X 第 3 页 §2.1」这样的引用不是渲染时猜出来的，是分块时就存下来的。

切分策略不是「每 N 个字切一刀」，而是：

  1. 先识别章节边界（优先用 PDF 自带的目录，其次用启发式）；
  2. 在章节内部按段落聚合到目标长度；
  3. 跨节不合并——一个块要么属于某一节，要么属于另一节。

跨节合并会让引用变成「§2 或 §3，不确定」，那种引用没有价值。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# 分块规则的版本号。**改动任何会影响块文本的东西，都要把它 +1。**
#
# 为什么需要它：索引重建是增量的，块一旦写进库，除非显式重跑就会一直用下去。
# 改了分块规则（切分方式、LaTeX 宏展开、公式/图注抽取……）却没升版本号，
# 结果是**新旧两种规则的块混在同一张表里**——检索质量悄悄变了，而任何地方
# 都不报错。等有人发现「怎么搜不到了」，已经无从判断是哪次改动造成的。
#
# 实测的例子：latex.py 的宏展开修好之前，\algname 这类自定义宏会被整段丢掉，
# 67 篇论文里的 14,837 处宏用法都受影响。那次修完之后，旧块仍然是坏的，
# 但没有任何信号提示它们需要重建。
#
# 版本号本身不触发重建，它只是让「哪些块是旧规则建的」变成一个**可以查的问题**。
# 见 indexer.stale_chunk_report()。
RULES_VERSION = 2
# 1 → 2：latex.py 改为花括号配对扫描展开自定义宏（旧的实现会吞掉方法名）。


def rules_meta() -> dict:
    """要写进 ``Chunk.meta`` 的构建信息。

    放在 meta（JSON）里而不是新增一个列：SQLite 下 ``create_all`` 不会改已有表，
    加列要写迁移；而这是个纯诊断字段，不值得为它引入迁移。
    """
    return {"rules": RULES_VERSION}

# 章节标题的常见形态：
#   "3 Method"、"3.2 Attention"、"IV. EXPERIMENTS"、"4.1.2. Details"
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:"
    r"(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})"          # 1 / 3.2 / 4.1.2
    r"|(?P<roman>[IVXLC]{1,6})\."                   # IV.
    r"|(?P<letter>[A-Z])\."                         # A.
    r")\s+(?P<title>\S.{0,80})$"
)

# 无编号但很常见的固定章节名
_KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "method",
    "methods", "methodology", "approach", "experiments", "experimental setup",
    "results", "discussion", "conclusion", "conclusions", "future work",
    "limitations", "references", "acknowledgments", "acknowledgements",
    "appendix", "摘要", "引言", "相关工作", "方法", "实验", "结果", "讨论",
    "结论", "参考文献", "致谢",
}

# 明显不是章节标题的行（页眉页脚、图表标题、公式编号）
_NOISE_LINE = re.compile(
    r"^\s*(?:"
    r"figure\s+\d|fig\.?\s*\d|table\s+\d|algorithm\s+\d|"
    r"\[\d+\]|\(\d+\)|"
    r"arxiv:\S+|https?://\S+|"
    r"\d{1,3}\s*$"                                   # 纯页码
    r")",
    re.I,
)


def estimate_tokens(text: str) -> int:
    """估算 token 数。

    不引入 tokenizer：真实的 BPE 分词器按模型而异（Claude、GPT、Qwen 各不同），
    为「切块大小」这一个用途引入一个几 MB 的依赖不划算。

    按字符类型分别估算，对中英混排足够准：
      * ASCII 字母：约 4 字符/token（英文的常见经验值）
      * CJK 字符：约 1 字符/token（中文基本一字一 token 到两字一 token）
      * 其它（数字、符号）：约 3 字符/token
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ")
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    other = len(text) - cjk - ascii_letters
    return int(cjk + ascii_letters / 4 + other / 3) + 1


@dataclass
class Section:
    """一个章节。"""

    title: str = ""
    level: int = 1
    path: str = ""            # 形如 "3 Method > 3.2 Attention"
    page_from: int = 0
    page_to: int = 0
    paragraphs: list[str] = field(default_factory=list)
    # 段落对应的页码，与 paragraphs 一一对应
    paragraph_pages: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass
class ChunkDraft:
    """待入库的分块。"""

    text: str
    section_path: str
    page_from: int
    page_to: int
    ord: int
    is_section_start: bool = False

    @property
    def n_tokens(self) -> int:
        return estimate_tokens(self.text)


# --------------------------------------------------------------------------
# 章节识别
# --------------------------------------------------------------------------


def _heading_level(line: str) -> tuple[int, str] | None:
    """判断一行是不是章节标题。返回 ``(层级, 标题)`` 或 None。"""
    stripped = line.strip()
    if not stripped or len(stripped) > 110:
        return None
    if _NOISE_LINE.match(stripped):
        return None

    match = _NUMBERED_HEADING.match(stripped)
    if match:
        num = match.group("num")
        if num:
            level = num.count(".") + 1
        elif match.group("roman"):
            level = 1
        else:
            level = 2
        title = match.group("title").strip()
        # 标题不该以句号结尾（那是句子不是标题）
        if title.endswith(".") and len(title.split()) > 8:
            return None
        return level, f"{num + ' ' if num else ''}{title}".strip()

    lowered = stripped.lower().rstrip(":")
    if lowered in _KNOWN_HEADINGS and len(stripped.split()) <= 6:
        return 1, stripped

    return None


def detect_sections(
    pages: list,  # list[PdfPage]
    outline: list[tuple[int, str, int]] | None = None,
) -> list[Section]:
    """把逐页文本切成章节。

    优先用 PDF 自带的目录（``outline``）——那是作者给的权威结构。
    没有目录时退回启发式：在每页里找像标题的行。
    """
    if outline:
        sections = _sections_from_outline(pages, outline)
        if len(sections) >= 2:
            return sections
        log.debug("PDF 目录信息不足（只得到 %d 节），改用启发式", len(sections))

    return _sections_heuristic(pages)


def _sections_from_outline(pages: list, outline: list[tuple[int, str, int]]) -> list[Section]:
    """按目录切分。目录给的是「标题 + 起始页」，所以每节的结束就是下一节的起始。"""
    entries = [(lvl, title, page) for lvl, title, page in outline if title and page > 0]
    if not entries:
        return []

    entries.sort(key=lambda item: item[2])
    sections: list[Section] = []
    parents: dict[int, str] = {}

    for index, (level, title, start_page) in enumerate(entries):
        next_page = entries[index + 1][2] if index + 1 < len(entries) else len(pages)
        if start_page > len(pages):
            continue

        parents[level] = title
        for deeper in [k for k in parents if k > level]:
            del parents[deeper]
        path = " > ".join(parents[k] for k in sorted(parents))

        text_pages = pages[start_page - 1 : max(start_page, next_page) - 1]
        section = Section(
            title=title,
            level=level,
            path=path,
            page_from=start_page,
            page_to=max(start_page, next_page - 1) if next_page > start_page else start_page,
        )
        for page in text_pages:
            for paragraph in _split_paragraphs(page.text):
                section.paragraphs.append(paragraph)
                section.paragraph_pages.append(page.number)
        if section.paragraphs:
            sections.append(section)

    return sections


def _sections_heuristic(pages: list) -> list[Section]:
    """没有目录时，靠标题行的形态来切分。"""
    sections: list[Section] = []
    current = Section(title="", level=0, path="", page_from=1, page_to=1)
    parents: dict[int, str] = {}

    for page in pages:
        for line in page.text.splitlines():
            heading = _heading_level(line)
            if heading is not None:
                level, title = heading
                if current.paragraphs:
                    sections.append(current)

                parents[level] = title
                for deeper in [k for k in parents if k > level]:
                    del parents[deeper]
                path = " > ".join(parents[k] for k in sorted(parents))

                current = Section(
                    title=title, level=level, path=path,
                    page_from=page.number, page_to=page.number,
                )
                continue

            stripped = line.strip()
            if stripped:
                current.paragraphs.append(stripped)
                current.paragraph_pages.append(page.number)
                current.page_to = page.number

        # 段落跨页时，同一个段落可能被拆开——这里按页重组一次
        current.paragraphs = _merge_wrapped(current.paragraphs)

    if current.paragraphs:
        sections.append(current)

    return sections or [Section(
        title="全文", level=0, path="全文",
        page_from=1, page_to=len(pages) or 1,
        paragraphs=_split_paragraphs("\n\n".join(p.text for p in pages)),
        paragraph_pages=[p.number for p in pages],
    )]


def _split_paragraphs(text: str) -> list[str]:
    """按空行切段；没有空行时按行切。"""
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    paragraphs = []
    for block in blocks:
        cleaned = re.sub(r"[ \t]+", " ", block).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _merge_wrapped(lines: list[str]) -> list[str]:
    """把被硬换行拆断的段落重新拼起来。

    PDF 提取出来的文本按视觉行断行，一个段落会被拆成很多行。
    判据是行尾有没有标点、下一行是不是大写开头——这是启发式，
    拼错的影响主要是检索时的匹配，不会丢内容。
    """
    if not lines:
        return []

    merged: list[str] = []
    buffer = lines[0]

    for line in lines[1:]:
        continues = (
            not buffer.rstrip().endswith((".", "。", "!", "！", "?", "？", ":", "：", ";", "；"))
            and not re.match(r"^\s*(?:[-•*·]|\d+[.)])\s", line)  # 不把列表项拼上去
            and len(buffer) > 20
        )
        if continues:
            buffer = f"{buffer} {line}"
        else:
            merged.append(buffer)
            buffer = line

    merged.append(buffer)

    # 太短的行多半是标题残留或页眉，单独成段没有意义，并入前一段
    result: list[str] = []
    for paragraph in merged:
        if len(paragraph) < 24 and result:
            result[-1] = f"{result[-1]} {paragraph}"
        else:
            result.append(paragraph)
    return result


# --------------------------------------------------------------------------
# 分块
# --------------------------------------------------------------------------


def chunk_sections(
    sections: list[Section],
    *,
    target_tokens: int = 800,
    overlap_tokens: int = 120,
    min_tokens: int = 40,
) -> list[ChunkDraft]:
    """把章节切成检索用的块。

    段落在块边界处**不切开**：一个段落是一个语义单元，从中间切开会让
    检索命中的半句话读起来没有意义。代价是块大小会有波动，
    但检索质量比块大小的整齐更重要。
    """
    drafts: list[ChunkDraft] = []
    order = 0

    for section in sections:
        if not section.paragraphs:
            continue

        buffer: list[str] = []
        buffer_tokens = 0
        buffer_pages: list[int] = []
        first_in_section = True

        # section 作为参数显式传入，而不是让闭包去捕获循环变量。
        # 闭包捕获的方式在当前调用顺序下也能跑对，但只要有人把 flush 的调用
        # 挪到循环外（或用生成器延迟执行），它就会静默地用上最后一轮的 section。
        def flush(current: Section) -> None:
            nonlocal buffer, buffer_tokens, buffer_pages, order, first_in_section
            if not buffer:
                return
            text = "\n\n".join(buffer)

            # 尾块太小就并入前一块——但只在同一节内。
            # 跨节合并会让引用定位变成「§2 或 §3」，那种引用没有价值。
            too_small = estimate_tokens(text) < min_tokens
            if too_small and drafts and not first_in_section and drafts[-1].section_path == (
                current.path or current.title
            ):
                drafts[-1].text += "\n\n" + text
                drafts[-1].page_to = max(drafts[-1].page_to, max(buffer_pages) if buffer_pages else 0)
                buffer, buffer_tokens, buffer_pages = [], 0, []
                return

            drafts.append(
                ChunkDraft(
                    text=text,
                    section_path=current.path or current.title,
                    page_from=min(buffer_pages) if buffer_pages else current.page_from,
                    page_to=max(buffer_pages) if buffer_pages else current.page_to,
                    ord=order,
                    is_section_start=first_in_section,
                )
            )
            order += 1
            first_in_section = False
            buffer, buffer_tokens, buffer_pages = [], 0, []

        for paragraph, page_number in zip(section.paragraphs, section.paragraph_pages, strict=False):
            tokens = estimate_tokens(paragraph)

            # 单段就超目标长度：它自己成块（不切段），否则会无限膨胀
            if tokens >= target_tokens * 1.6:
                flush(section)
                buffer = [paragraph]
                buffer_pages = [page_number]
                buffer_tokens = tokens
                flush(section)
                continue

            if buffer_tokens + tokens > target_tokens and buffer:
                flush(section)
                # 重叠：把上一块的结尾带进新块，避免跨块的问题被切断
                if overlap_tokens > 0 and drafts:
                    tail = _tail_by_tokens(drafts[-1].text, overlap_tokens)
                    if tail:
                        buffer = [tail]
                        buffer_pages = [drafts[-1].page_to]
                        buffer_tokens = estimate_tokens(tail)

            buffer.append(paragraph)
            buffer_pages.append(page_number)
            buffer_tokens += tokens

        flush(section)

    return drafts


def _tail_by_tokens(text: str, tokens: int) -> str:
    """按 token 估算截取文本尾部。"""
    if not text or tokens <= 0:
        return ""
    ratio = tokens / max(1, estimate_tokens(text))
    if ratio >= 1:
        return ""
    cut = int(len(text) * ratio)
    tail = text[-cut:] if cut else ""
    # 从句子边界开始，避免以半句话开头
    for separator in ("。", ". ", "\n"):
        index = tail.find(separator)
        if 0 <= index < len(tail) // 2:
            return tail[index + len(separator):].strip()
    return tail.strip()


@dataclass
class NoteSection:
    """笔记里的一节。"""

    heading: str          # 小节标题，同时用作引用定位符
    text: str
    level: int = 2

    @property
    def n_tokens(self) -> int:
        return estimate_tokens(self.text)


# 小节小于这个 token 数就和相邻节合并。
#
# 笔记里「待跟进」这类小节常常只有一行字。单独成块的话，它在检索里是一个
# 极短的高分项（bm25 对短文档天然给高分），会把真正有内容的节挤下去。
_MIN_NOTE_SECTION_TOKENS = 120


def chunk_markdown(
    text: str,
    *,
    max_tokens: int = 800,
    min_tokens: int = _MIN_NOTE_SECTION_TOKENS,
) -> list[NoteSection]:
    """把 Markdown 笔记按小节切开。

    和论文的分块逻辑分开写，因为两者的结构保证不同：论文靠 PDF 目录或
    编号标题，得猜；笔记的 ``##`` 是我们自己生成的，结构是确定的。
    共用一套启发式规则反而会把简单的事做复杂。

    规则：
      * 在 ``##`` / ``###`` 处切；``###`` 归入它所属的 ``##``；
      * 首个标题之前的内容单独成节，叫「前言」；
      * 太短的节向后合并，避免产生一堆只有一行的高分小块；
      * 超长的节按段落续切，不跨节合并。

    标题要保留在块的正文里：引用定位符是标题，正文里再出现一次，
    检索「方法」这类词时这一节才排得上去。
    """
    if not text or not text.strip():
        return []

    heading_re = re.compile(r"^(#{2,3})\s+(.+?)\s*$")

    # 先按 ## 切成若干节（### 及更深的内容跟着它的父节走）
    sections: list[tuple[str, int, list[str]]] = []
    preamble: list[str] = []
    current: tuple[str, int, list[str]] | None = None

    for line in text.splitlines():
        match = heading_re.match(line)
        if match and len(match.group(1)) == 2:
            if current is not None:
                sections.append(current)
            elif preamble and any(p.strip() for p in preamble):
                sections.append(("前言", 2, preamble))
                preamble = []
            current = (match.group(2), 2, [line])
            continue
        if current is None:
            preamble.append(line)
        else:
            current[2].append(line)
    if current is not None:
        sections.append(current)
    elif preamble and any(p.strip() for p in preamble):
        sections.append(("前言", 2, preamble))

    # 过短的节向后合并。合并时保留各自的标题行，不然小节名就丢了
    merged: list[tuple[str, int, list[str]]] = []
    for heading, level, lines in sections:
        body = "\n".join(lines).strip()
        if merged and estimate_tokens(body) < min_tokens:
            prev_heading, prev_level, prev_lines = merged[-1]
            merged[-1] = (prev_heading, prev_level, [*prev_lines, "", body])
            continue
        merged.append((heading, level, lines))
    # 末节也可能太短，这时并进前一节
    if len(merged) > 1 and estimate_tokens("\n".join(merged[-1][2])) < min_tokens:
        heading, level, lines = merged.pop()
        prev_heading, prev_level, prev_lines = merged[-1]
        merged[-1] = (prev_heading, prev_level, [*prev_lines, "", "\n".join(lines).strip()])

    # 超长的节按段落续切
    out: list[NoteSection] = []
    for heading, level, lines in merged:
        body = "\n".join(lines).strip()
        if estimate_tokens(body) <= max_tokens:
            out.append(NoteSection(heading=heading, text=body, level=level))
            continue
        chunk: list[str] = []
        for paragraph in body.split("\n\n"):
            joined = estimate_tokens("\n\n".join([*chunk, paragraph]))
            if joined > max_tokens and estimate_tokens("\n\n".join(chunk)) >= min_tokens:
                out.append(NoteSection(heading=heading, text="\n\n".join(chunk), level=level))
                # 续块保留标题，检索时才知道它属于哪一节
                chunk = [f"（{heading} 续）", paragraph]
            else:
                # 注意这里**不**切：已经攒下的部分还太小就切开，会得到一个
                # 只有标题的碎块——实测「关键技术」那一节就是因为紧跟的段落
                # 太长，被切出一个 6 token 的块。超长的一块好过一个碎片：
                # 碎片在 bm25 里是极短文档，天然拿高分，会盖过真正有内容的节。
                chunk.append(paragraph)
        if chunk:
            out.append(NoteSection(heading=heading, text="\n\n".join(chunk), level=level))
    return out


def chunk_document(
    pages: list,
    outline: list[tuple[int, str, int]] | None = None,
    *,
    target_tokens: int = 800,
    overlap_tokens: int = 120,
) -> tuple[list[Section], list[ChunkDraft]]:
    """完整流程：识别章节 -> 分块。返回两者以便调用方查看/调试。"""
    sections = detect_sections(pages, outline)
    drafts = chunk_sections(sections, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    log.debug("分块完成：%d 节 -> %d 块", len(sections), len(drafts))
    return sections, drafts


__all__ = [
    "RULES_VERSION",
    "ChunkDraft",
    "NoteSection",
    "Section",
    "chunk_document",
    "chunk_markdown",
    "chunk_sections",
    "detect_sections",
    "estimate_tokens",
    "rules_meta",
]
