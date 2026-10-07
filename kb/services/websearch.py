"""联网检索：给知识库补上「库里没有的东西」。

**为什么要分后端而不是接一个「网页搜索」。** 这个项目是论文知识库，
它的用户问题和通用搜索引擎的用户问题不一样：「这个方向还有哪些新工作」
「这篇被谁引用了」「这个损失函数在别人仓库里怎么写」——这三类问题分别有
专门的、免费的、结构化的接口，比抓搜索结果页可靠得多，也快得多。
通用网页搜索只在查官方文档、issue 讨论这类东西时才有优势，而它需要 API Key。

**实测过的网络情况**（决定了默认接哪些）：OpenAlex、arXiv、GitHub 免 Key 可用；
Tavily 域名可达但要 Key；DuckDuckGo / SearXNG / Brave 的 TLS 握手直接超时
（与 cdn.jsdelivr.net 被墙是同一现象）；Wikipedia 与 Google CSE 对数据中心 IP
返回 403。所以「抓 DDG 结果页」这条常见路子在这个环境里根本走不通，
不该把它当作默认实现。

每个后端独立失败：一个超时不影响其它的。全部失败时返回明确的错误，
而不是静默返回空——「没搜到」和「搜索坏了」对用户是两件事。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "kb-knowledge-base/0.1 (personal research manager)"

# 各后端的超时都压得比较短：联网是给对话补充信息，不是主体，
# 让用户为一个可选步骤等半分钟是不划算的。
SEARCH_TIMEOUT = 12.0
FETCH_TIMEOUT = 15.0
MAX_PAGE_BYTES = 2 * 1024 * 1024


@dataclass
class WebResult:
    """一条联网检索结果。"""

    title: str
    url: str
    snippet: str = ""
    source: str = ""            # 后端名：openalex / github / tavily …
    kind: str = "web"           # paper / code / web
    published: str = ""
    authors: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    content: str = ""           # 抓取的正文（可能为空）

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "kind": self.kind,
            "published": self.published,
            "authors": self.authors,
        }

    def as_context(self, marker: str) -> str:
        """给模型看的文本形式。"""
        bits = [f"[{marker}] {self.title}"]
        if self.authors:
            bits.append("作者：" + "、".join(self.authors[:4]))
        if self.published:
            bits.append(f"时间：{self.published}")
        bits.append(f"来源：{self.url}")
        body = self.content or self.snippet
        if body:
            bits.append("")
            bits.append(body.strip())
        return "\n".join(bits)


# --------------------------------------------------------------------------
# 后端
# --------------------------------------------------------------------------


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=SEARCH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def search_openalex(query: str, limit: int) -> list[WebResult]:
    """OpenAlex：开放学术图谱，免 Key。

    强项是**引用关系**与**领域全景**——「这个方向最近有什么」这类问题
    靠它比靠全文检索靠谱得多。
    """
    with _client() as client:
        response = client.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": limit,
                # 只要标题/摘要/DOI/年份，不要全文——体积小很多
                "select": "id,doi,title,publication_year,authorships,primary_location,abstract_inverted_index",
            },
        )
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in payload.get("results", [])[:limit]:
        url = item.get("doi") or item.get("id") or ""
        location = item.get("primary_location") or {}
        title = item.get("title") or "（无标题）"

        # OpenAlex 的摘要是倒排索引，要还原成文本
        snippet = _inverted_to_text(item.get("abstract_inverted_index"))[:600]

        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (item.get("authorships") or [])[:5]
        ]
        results.append(
            WebResult(
                title=title,
                url=url,
                snippet=snippet,
                source="openalex",
                kind="paper",
                published=str(item.get("publication_year") or ""),
                authors=[a for a in authors if a],
                extra={
                    "venue": (location.get("source") or {}).get("display_name", ""),
                    "cited_by": item.get("cited_by_count"),
                },
            )
        )
    return results


def _inverted_to_text(inverted: dict | None) -> str:
    """把 OpenAlex 的倒排索引还原成摘要文本。"""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        for index in indexes:
            positions.append((index, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def search_arxiv(query: str, limit: int) -> list[WebResult]:
    """arXiv：预印本。新工作往往先出现在这里，比正式发表早半年到一年。"""
    import defusedxml.ElementTree as ET

    with _client() as client:
        response = client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results": limit,
                "sortBy": "relevance",
            },
        )
        response.raise_for_status()
        text = response.text

    results = []
    try:
        root = ET.fromstring(text)
    except Exception as exc:
        log.debug("arXiv 返回内容无法解析：%s", exc)
        return []

    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns)[:limit]:
        title = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
        url = (entry.findtext("a:id", "", ns) or "").strip()
        published = (entry.findtext("a:published", "", ns) or "")[:10]
        authors = [
            (a.findtext("a:name", "", ns) or "").strip()
            for a in entry.findall("a:author", ns)[:5]
        ]
        results.append(
            WebResult(
                title=re.sub(r"\s+", " ", title),
                url=url,
                snippet=re.sub(r"\s+", " ", summary)[:600],
                source="arxiv",
                kind="paper",
                published=published,
                authors=[a for a in authors if a],
            )
        )
    return results


def search_semantic_scholar(query: str, limit: int) -> list[WebResult]:
    """Semantic Scholar：摘要 + 引用数。免费额度低，被限流是常态。"""
    with _client() as client:
        response = client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,url,year,authors,citationCount,venue",
            },
        )
        if response.status_code == 429:
            # 限流不算错误：这是它的常态，别的后端照样能出结果
            log.debug("Semantic Scholar 限流，跳过")
            return []
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in payload.get("data", [])[:limit]:
        results.append(
            WebResult(
                title=item.get("title") or "（无标题）",
                url=item.get("url") or "",
                snippet=(item.get("abstract") or "")[:600],
                source="semantic_scholar",
                kind="paper",
                published=str(item.get("year") or ""),
                authors=[a.get("name", "") for a in (item.get("authors") or [])[:5]],
                extra={
                    "venue": item.get("venue"),
                    "cited_by": item.get("citationCount"),
                },
            )
        )
    return results


def search_github(query: str, limit: int) -> list[WebResult]:
    """GitHub 仓库搜索：查某个方法在别人仓库里怎么实现的。"""
    with _client() as client:
        response = client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": limit, "sort": "stars"},
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status_code == 403:
            log.debug("GitHub 搜索被限流")
            return []
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in payload.get("items", [])[:limit]:
        results.append(
            WebResult(
                title=item.get("full_name") or "",
                url=item.get("html_url") or "",
                snippet=(item.get("description") or "")[:400],
                source="github",
                kind="code",
                published=(item.get("updated_at") or "")[:10],
                extra={
                    "stars": item.get("stargazers_count"),
                    "language": item.get("language"),
                },
            )
        )
    return results


def search_tavily(query: str, limit: int, api_key: str) -> list[WebResult]:
    """Tavily：通用网页搜索，需要 Key。

    没有 Key 时这个后端直接不可用——**不静默跳过**，调用方要能告诉用户
    「通用网页搜索没配 Key，学术和代码检索仍然可用」。
    """
    if not api_key:
        return []
    with _client() as client:
        response = client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
            },
        )
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in payload.get("results", [])[:limit]:
        results.append(
            WebResult(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=(item.get("content") or "")[:600],
                source="tavily",
                kind="web",
                published=(item.get("published_date") or "")[:10],
            )
        )
    return results


# --------------------------------------------------------------------------
# 正文抓取
# --------------------------------------------------------------------------


def fetch_page(url: str, *, max_chars: int = 4000) -> str:
    """抓取网页并抽取正文。

    用 ``trafilatura`` 而不是直接把 HTML 塞给模型：一个新闻页的 HTML
    动辄几百 KB，其中正文可能只有几 KB，剩下的都是导航、广告和脚本。
    不抽取的话上下文会被垃圾占满，而且模型还得自己在噪声里找正文。
    """
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                return ""
            chunks = []
            size = 0
            for chunk in response.iter_bytes(64 * 1024):
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_PAGE_BYTES:
                    break
            html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
    except Exception as exc:
        log.debug("抓取 %s 失败：%s", url, exc)
        return ""

    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
    except Exception:
        text = None

    if not text:
        # trafilatura 抽不出来时退回粗暴去标签——总比什么都没有好
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)

    return (text or "").strip()[:max_chars]


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def search(
    query: str,
    *,
    settings,
    limit: int | None = None,
    fetch_pages: bool | None = None,
    kinds: tuple[str, ...] = ("paper", "code", "web"),
) -> tuple[list[WebResult], list[str]]:
    """联网检索，返回 (结果, 出错的后端说明)。

    **后端各自独立失败**：某个源超时或被限流时，其余的照常返回。
    只有全部失败才把 errors 汇总出来——「没搜到」和「搜索全挂了」
    对用户是两件事，不能都表现为一个空列表。
    """
    query = (query or "").strip()
    if not query:
        return [], []

    limit = limit or int(settings.get("websearch.max_results"))
    if fetch_pages is None:
        fetch_pages = bool(settings.get("websearch.fetch_pages"))

    # 每个后端分一小份配额，避免一个源独吞
    per_source = max(2, limit // 2)
    results: list[WebResult] = []
    errors: list[str] = []

    backends: list[tuple[str, Any]] = []
    if "paper" in kinds and settings.get("websearch.academic"):
        backends.append(("openalex", lambda: search_openalex(query, per_source)))
        backends.append(("arxiv", lambda: search_arxiv(query, per_source)))
        backends.append(("semantic_scholar", lambda: search_semantic_scholar(query, per_source)))
    if "code" in kinds and settings.get("websearch.code"):
        backends.append(("github", lambda: search_github(query, per_source)))
    if "web" in kinds:
        key = settings.get("websearch.tavily_api_key") or ""
        if key:
            backends.append(("tavily", lambda: search_tavily(query, per_source, key)))
        else:
            errors.append("通用网页搜索未启用（未配置 Tavily API Key）")

    for name, run in backends:
        try:
            found = run()
            results.extend(found)
            log.debug("联网检索 %s：%d 条", name, len(found))
        except Exception as exc:
            errors.append(f"{name} 检索失败：{type(exc).__name__}")
            log.debug("联网检索 %s 失败", name, exc_info=True)

    # 去重：不同源可能返回同一篇论文（DOI 或标题相同）
    deduped: list[WebResult] = []
    seen: set[str] = set()
    for item in results:
        key = (item.url or item.title).lower().rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # 论文类结果按引用数排前面——被引多的通常更值得先看
    deduped.sort(
        key=lambda r: (r.kind != "paper", -(r.extra.get("cited_by") or 0), r.title),
    )
    chosen = deduped[:limit]

    if fetch_pages:
        max_chars = int(settings.get("websearch.max_page_chars"))
        for item in chosen:
            # 已经有摘要且够长的就不抓了——抓取是这里最慢的一步
            if len(item.snippet) > 800:
                continue
            content = fetch_page(item.url, max_chars=max_chars)
            if content:
                item.content = content

    return chosen, errors


__all__ = ["WebResult", "fetch_page", "search"]
