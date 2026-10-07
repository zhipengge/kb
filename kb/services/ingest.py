"""论文入库：从 arXiv、链接、或仅有标题的信息找到并下载论文。

这是需求里「通过链接、实体 PDF 论文文件等方式增加新的论文」的实现。

难点不在下载，而在**把一个模糊的标题变成一个确定的论文**。用户给的往往
是文件名（``UniAD-Planning-oriented Autonomous Driving.pdf``）或随手写的
标题，而 arXiv 的标题检索是精确匹配——直接拿整串去查，十有八九查不到。
所以这里的做法是：逐步放宽查询，再用相似度把结果与原始输入比对，
取最像的那一个；相似度不够就**不猜**，报「没找到」让人来确认。

最后一点很重要：自动猜错会把两篇不同的论文混成一条记录，
而且是在用户不知情的情况下。
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# 用 defusedxml 而不是标准库的 ElementTree：arXiv 返回的 XML 是**外部输入**，
# 标准库的解析器对「十亿笑声」这类实体展开攻击没有防护，
# 一个精心构造的响应就能把内存吃光。
from defusedxml import ElementTree as ET

from .paths import sanitize_filename

log = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}

# arXiv 官方要求请求间隔约 3 秒。这不是建议——被限流后整个功能都不可用。
MIN_REQUEST_INTERVAL = 3.0
_last_request = 0.0


class IngestError(RuntimeError):
    """入库失败。消息面向用户。"""


def _polite(
    url: str,
    *,
    timeout: float = 60.0,
    total_timeout: float = 240.0,
    max_bytes: int = 128 * 1024 * 1024,
):
    """带速率限制的 GET。

    ``timeout`` 是**单次 socket 操作**的超时，不是整个请求的时长。
    服务端若以极慢的速度滴数据（限流时常见），它永远不触发，下载会
    无限期挂着——表现为批量任务静默卡死，不报错、不占 CPU。
    所以这里额外用一个总时限兜底。
    """
    global _last_request
    elapsed = time.monotonic() - _last_request
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    # S310 告警的是「URL 可能指向 file: 或自定义协议」。这里的防护是显式的：
    # 协议前缀必须先通过校验，而所有调用点的前缀都是本模块写死的常量，
    # 只有路径部分（arXiv 编号）来自外部，且已由 resolve_arxiv 归一化。
    if not url.startswith(("http://", "https://")):
        return None, f"拒绝非 http(s) 地址：{url[:40]}"

    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "kb-knowledge-base/0.1 (personal research manager)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            _last_request = time.monotonic()
            deadline = time.monotonic() + total_timeout
            # 分块读而不是一次 read(max_bytes)：一次读完就没机会检查总时限
            chunks: list[bytes] = []
            remaining = max_bytes
            while remaining > 0:
                if time.monotonic() > deadline:
                    got = sum(len(c) for c in chunks)
                    return None, (
                        f"下载超过 {total_timeout:.0f} 秒仍未完成"
                        f"（已收到 {got // 1024} KB），判定为被限流，已放弃"
                    )
                chunk = response.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks), None
    except Exception as exc:
        _last_request = time.monotonic()
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# 标题清理与匹配
# --------------------------------------------------------------------------


# 文件名里常见的「短名-标题」结构：UniAD-Planning-oriented...、BEVWorld- A Multimodal...
_SHORTNAME_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+._-]{1,24}\s*[-–—:]\s*")

# 去掉版本、下载后缀之类
_NOISE_SUFFIX = re.compile(
    r"[\s_-]*(?:v\d+|final|preprint|draft|camera[- ]ready|arxiv[-\s]?\d{4}\.\d{4,5})\s*$",
    re.I,
)


def _ascii_fold(text: str) -> str:
    """把变音字符折成 ASCII（Bézier → Bezier）。

    arXiv 的标题检索对非 ASCII 字符很不友好：``Bézier`` 直接查是 0 条结果，
    换成 ``Bezier`` 就能查到。论文标题里带变音符号的不少（Bézier、Schrödinger、
    Müller 之类），不折叠的话这些论文永远解析不出来。
    """
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # 少数字符 NFKD 折不出来，单独处理
    for source, target in (("ø", "o"), ("Ø", "O"), ("ß", "ss"), ("æ", "ae"), ("œ", "oe")):
        folded = folded.replace(source, target)
    return folded


def clean_title(raw: str) -> list[str]:
    """把文件名/随手写的标题变成若干候选查询串。

    返回**多个**候选而不是一个：从最严格到最宽松依次尝试，
    比一开始就用最宽松的查询要准——宽松查询很容易命中不相关的论文。
    """
    text = Path(str(raw)).stem if str(raw).lower().endswith(".pdf") else str(raw)
    text = text.replace("_", " ").strip()

    candidates: list[str] = []

    # 去掉 arXiv 编号前缀（arxiv-2401.01339 → 空）
    if re.match(r"^arxiv[-\s]?\d{4}\.\d{4,5}$", text.strip(), re.I):
        return [text.strip()]

    # 完整标题
    candidates.append(text)

    # 去掉「短名-」前缀
    stripped = _SHORTNAME_PREFIX.sub("", text).strip()
    if stripped and stripped != text:
        candidates.append(stripped)

    # 去掉尾部噪音
    for value in list(candidates):
        cleaned = _NOISE_SUFFIX.sub("", value).strip(" .-–—")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    # 最后退到「主标题」——去掉冒号/破折号后面的副标题
    for value in list(candidates):
        for separator in ("–", "—", ":"):
            if separator in value:
                head = value.split(separator)[0].strip()
                if len(head) >= 12 and head not in candidates:
                    candidates.append(head)

    return [c for c in candidates if len(c) >= 4]


def _normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def similarity(left: str, right: str) -> float:
    """两个标题的相似度（0-100）。"""
    try:
        from rapidfuzz import fuzz
    except ImportError:  # pragma: no cover
        return 100.0 if _normalize(left) == _normalize(right) else 0.0
    return float(fuzz.token_set_ratio(_normalize(left), _normalize(right)))


# --------------------------------------------------------------------------
# arXiv 检索
# --------------------------------------------------------------------------


@dataclass
class ArxivCandidate:
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    summary: str = ""
    published: str = ""
    categories: list[str] = field(default_factory=list)
    pdf_url: str = ""

    @property
    def versionless_id(self) -> str:
        return re.sub(r"v\d+$", "", self.arxiv_id)


def search_arxiv(query: str, *, limit: int = 5, field: str = "ti") -> list[ArxivCandidate]:
    """检索 arXiv。

    ``field`` 为 ``ti``（标题）、``all``（全文）或 ``raw``（原样使用 query）。

    必须支持 ``raw``：像 ``id:2401.01339`` 这种查询本身已经限定了字段，
    再套一层 ``all:"…"`` 会变成 ``all:"id:2401.01339"``，
    于是把一个精确的 ID 查询变成了「找包含这个字符串的文档」，什么都查不到。
    """
    expression = query if field == "raw" else f'{field}:"{query}"'
    params = {
        "search_query": expression,
        "max_results": str(limit),
        "sortBy": "relevance",
    }

    # 重试一次。arXiv 偶发连接失败或返回残缺响应，而这类失败一旦被当成
    # 「查不到」，结果是**论文永远解析不出来**——批量处理时尤其可惜：
    # 一次抖动就让一篇论文白白漏掉，而且失败信息看起来像「arXiv 上没有」。
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    data, error = _polite(url)
    if data is None:
        log.debug("arXiv 检索失败，重试一次：%s", error)
        time.sleep(2.0)
        data, error = _polite(url)
    if data is None:
        raise IngestError(f"arXiv 检索失败：{error}")

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise IngestError(f"arXiv 返回的内容无法解析：{exc}") from exc

    results: list[ArxivCandidate] = []
    for entry in root.findall("a:entry", _ATOM):
        identifier = (entry.findtext("a:id", "", _ATOM) or "").split("/abs/")[-1]
        if not identifier:
            continue
        title = " ".join((entry.findtext("a:title", "", _ATOM) or "").split())
        summary = " ".join((entry.findtext("a:summary", "", _ATOM) or "").split())
        authors = [
            (node.findtext("a:name", "", _ATOM) or "").strip()
            for node in entry.findall("a:author", _ATOM)
        ]
        categories = [
            node.get("term", "")
            for node in entry.findall("a:category", _ATOM)
            if node.get("term")
        ]
        pdf_url = ""
        for link in entry.findall("a:link", _ATOM):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        results.append(
            ArxivCandidate(
                arxiv_id=identifier,
                title=title,
                authors=[a for a in authors if a],
                summary=summary,
                published=entry.findtext("a:published", "", _ATOM) or "",
                categories=categories,
                pdf_url=pdf_url,
            )
        )
    return results


def _extract_shortname(raw: str) -> str | None:
    """从「短名-标题」形式的输入里取出短名。

    论文的项目名通常写在文件名最前面（``SplatAD-Real-Time Lidar...``）。
    它往往也是全篇最具辨识度的单个词，适合作为最后的检索兜底。

    只接受 3-25 个字符、纯字母数字（可含内部连字符）的片段——
    再短会命中大量无关内容，再长就不是项目名了。
    """
    text = Path(str(raw)).stem if str(raw).lower().endswith(".pdf") else str(raw)
    text = text.strip()

    # 短名后跟连字符或冒号
    match = re.match(r"^([A-Za-z][A-Za-z0-9]{2,24})\s*[-–—:]\s*\S", text)
    if match:
        return match.group(1)
    # 也可能整串就是一个词（BETAV.pdf）
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{2,24}", text):
        return text
    return None


def resolve_arxiv(
    raw_title: str, *, min_similarity: float = 78.0
) -> tuple[ArxivCandidate | None, str]:
    """把一个模糊标题解析成确定的 arXiv 论文。

    返回 ``(候选, 说明)``。相似度不够时返回 ``(None, 原因)``——
    **宁可报「没找到」，也不要猜**。猜错的代价是把两篇不同的论文
    混成一条记录，而且用户不会察觉。
    """
    candidates = clean_title(raw_title)

    # 输入本身带 arXiv 编号：直接用 id: 查询，这是最确定的路径
    match = re.search(r"(\d{4}\.\d{4,5})", raw_title)
    if match:
        try:
            found = search_arxiv(f"id:{match.group(1)}", limit=1, field="raw")
            if found:
                return found[0], "按 arXiv 编号直接命中"
        except IngestError as exc:
            log.warning("按编号查询 %s 失败：%s", match.group(1), exc)

    best: tuple[float, ArxivCandidate] | None = None
    tried: list[str] = []

    # 查询从严格到宽松。顺序有讲究：先精确标题，再短前缀，再全文短语，
    # 最后才用关键词——越宽松越容易命中不相关的论文。
    queries: list[tuple[str, str]] = []
    for query in candidates[:3]:
        queries.append((query, "ti"))
        # 同时试去变音符号的版本。Bézier 这类字符会让 arXiv 标题检索零结果，
        # 而去掉之后完全正常。
        folded = _ascii_fold(query)
        if folded != query:
            queries.append((folded, "ti"))

    # 长标题往往整串查不到，但前几个词能查到——arXiv 的标题匹配
    # 对超长查询词相当敏感。用前缀再试一轮。
    for query in candidates[:2]:
        words = query.split()
        if len(words) > 7:
            prefix = " ".join(words[:7])
            queries.append((prefix, "ti"))
            folded = _ascii_fold(prefix)
            if folded != prefix:
                queries.append((folded, "ti"))

    for query in candidates[:2]:
        queries.append((query, "all"))

    # 文件名被截断时（常见：xxx-autonomous-drivi.pdf）精确标题匹配不上，
    # 退到关键词检索。
    #
    # 用 AND 连接的关键词，**不是短语**：文件名末尾常被截断成半个词
    # （"drivi"），短语匹配要求逐字相符，带着这半个词就一条都搜不到。
    # 拆成 `all:a AND all:b` 之后，半个词至多让这一项匹配不上，其余仍然有效。
    for query in candidates[:1]:
        words = [w for w in re.findall(r"[A-Za-z]{4,}", query)][:8]
        # 末尾那个词很可能是被截断的，最不可靠——先去掉它再试
        if len(words) > 3 and len(words[-1]) >= 4:
            expression = " AND ".join(f"all:{w}" for w in words[:-1])
            queries.append((expression, "raw"))
        if len(words) > 2:
            expression = " AND ".join(f"all:{w}" for w in words)
            queries.append((expression, "raw"))

    # 最后一路：只查「短名」这一个词。
    #
    # 论文的项目名（SplatAD、DriveMA 之类）在标题、摘要、代码库里反复出现，
    # 是整篇论文里最具辨识度的单个 token；而把长标题当查询反而查不到——
    # arXiv 对长查询的处理很不稳定（实测 all:"SplatAD lidar camera
    # rendering" 零结果，all:SplatAD 一下就中）。所以留这一招兜底。
    shortname = _extract_shortname(raw_title)
    if shortname:
        queries.append((f"all:{shortname}", "raw"))

    for query, field_name in queries:
        if len(query) < 6:
            continue
        tried.append(f"{field_name}:{query[:38]}")
        try:
            results = search_arxiv(query, limit=8, field=field_name)
        except IngestError as exc:
            log.warning("检索 %r 失败：%s", query, exc)
            continue

        for candidate in results:
            score = max(
                similarity(raw_title, candidate.title),
                similarity(query, candidate.title),
            )
            if best is None or score > best[0]:
                best = (score, candidate)

        if best is not None and best[0] >= 95:
            break

    if best is None:
        return None, f"arXiv 上没有找到（尝试过：{'；'.join(tried[:4])}）"

    score, candidate = best
    if score < min_similarity:
        return None, (
            f"最接近的是「{candidate.title[:60]}」（相似度 {score:.0f}，"
            f"低于阈值 {min_similarity:.0f}），未自动采用"
        )
    return candidate, f"匹配到「{candidate.title[:60]}」（相似度 {score:.0f}）"


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------


def download_pdf(arxiv_id: str, target_dir: Path, *, filename: str | None = None) -> tuple[Path | None, str]:
    """从 arXiv 下载 PDF。"""
    identifier = re.sub(r"v\d+$", "", arxiv_id.strip())
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    name = sanitize_filename(filename or f"{identifier}.pdf", fallback=f"{identifier}.pdf")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    target = target_dir / name

    if target.is_file() and target.stat().st_size > 10_000:
        return target, "已存在，跳过下载"

    data, error = _polite(ARXIV_PDF.format(arxiv_id=identifier))
    if data is None:
        return None, f"下载失败：{error}"
    if data[:5] != b"%PDF-":
        return None, "下载到的不是 PDF（可能该论文没有 PDF，或触发了限流）"

    # 先写临时文件再改名，避免中断时留下半个文件被判为「已存在」
    tmp = target.with_suffix(".pdf.part")
    try:
        tmp.write_bytes(data)
        tmp.replace(target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return None, f"写入失败：{exc}"

    return target, f"{len(data) / 1024:.0f} KB"


__all__ = [
    "ArxivCandidate",
    "IngestError",
    "clean_title",
    "download_pdf",
    "resolve_arxiv",
    "search_arxiv",
    "similarity",
]
