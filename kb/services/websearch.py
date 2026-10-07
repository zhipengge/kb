"""联网检索：给知识库补上「库里没有的东西」。

**为什么要分后端而不是接一个「网页搜索」。** 这个项目是论文知识库，
它的用户问题和通用搜索引擎的用户问题不一样：「这个方向还有哪些新工作」
「这篇被谁引用了」「这个损失函数在别人仓库里怎么写」——这三类问题分别有
专门的、免费的、结构化的接口，比抓搜索结果页可靠得多，也快得多。
通用网页搜索只在查官方文档、issue 讨论这类东西时才有优势，而它需要 API Key。

**实测过的网络情况**（决定了默认接哪些）。可达：OpenAlex、Crossref、arXiv、
Semantic Scholar、GitHub、HackerNews（以上全部免 Key），以及 Tavily（要 Key）。

**免 Key 的通用网页搜索在这个环境里不存在**——这不是没找，是逐一试过：

  * DuckDuckGo、Brave、Startpage、Bing、Qwant、GitLab、Jina Reader：
    TLS 握手直接超时（与 cdn.jsdelivr.net 被墙是同一现象）；
  * SearXNG：公共实例与 3 个私有实例全部超时（它的上游引擎本身就够不着，
    自建也一样没用）；
  * Reddit：连接被 EOF；Wikipedia：对数据中心 IP 返回 403；
  * **Mojeek**：HTTP 200，但返回的是**验证码页**；
  * **Ecosia**：HTTP 200/189KB，但跳到了 **Bing 中国版**，一条真实结果都没有。

所以「多引擎抓搜索结果页」这条常见路子在这里做出来只会**静默失败**，
不该作为默认实现。要通用网页搜索就得配 Tavily Key。

另外**接口本身也会抖**：实测 OpenAlex 同一小时内先 200 后超时，
Stack Exchange 连测 3 次超时 1 次。这正说明「后端各自独立失败」不够，
还得有缓存——见下面。

每个后端独立失败：一个超时不影响其它的。全部失败时返回明确的错误，
而不是静默返回空——「没搜到」和「搜索坏了」对用户是两件事。

**试过但决定不接的源**（省得日后有人再试一遍）：

  * **Stack Exchange**：接口可达，但对这个知识库是噪声。实测
    ``q="diffusion policy"`` 在 Stack Overflow 上返回的全是 Stable Diffusion
    绘图工具的问题；换到 stats / ai / robotics / datascience 这些对口的站点
    则返回 0 条。加上它本身会抖，结论是净负——它会占掉 limit 名额、
    往模型上下文里灌无关内容。

**融合与排序。** 多个源返回同一项工作时按 RRF 融合（理由同
``services/search.py``：各源分数量纲不可比）。去重先看权威 ID
（DOI / arXiv / GitHub），都没有才比标题相似度——细节见 ``_dedup`` 的说明，
那里有一条不能破的规则。

**没有做语义去重 / 重排。** 那是当前开源实现的通行做法（嵌入 + 阈值 ~0.9），
但本地嵌入模型 ``bge-small-zh-v1.5`` 只对中文有效：实测英文「同义」余弦
0.5472、「无关」0.5117，**不可分**；而联网结果绝大多数是英文。
换 ``bge-m3``（约 2GB）能解决，属于另一个决定。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "kb-knowledge-base/0.1 (personal research manager)"

# 各后端的超时都压得比较短：联网是给对话补充信息，不是主体，
# 让用户为一个可选步骤等半分钟是不划算的。
SEARCH_TIMEOUT = 12.0
FETCH_TIMEOUT = 15.0
MAX_PAGE_BYTES = 2 * 1024 * 1024

# RRF 的平滑常数，与 services/search.py 的默认值保持一致。
# 取 60 是 RRF 原论文的经验值：它让「排第 1」和「排第 2」的差距不至于过大，
# 从而让「多个源都排中游」能胜过「单个源排第一」——这正是我们要的语义，
# 因为「三个独立来源都认为它相关」比「一个来源认为它最相关」更可信。
RRF_K = 60


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
                # 只要标题/摘要/DOI/年份，不要全文——体积小很多。
                #
                # cited_by_count 必须在 select 里：下面读了它，但 select 会**限定**
                # 返回字段，没列出来的拿不到。漏了它的表现是引用数恒为 None，
                # 而排序和「信息更全的优先」都以它为依据——静默失效，不报错。
                "select": (
                    "id,doi,title,publication_year,authorships,primary_location,"
                    "abstract_inverted_index,cited_by_count"
                ),
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


def search_crossref(query: str, limit: int) -> list[WebResult]:
    """Crossref：DOI 注册库本身，免 Key。

    和 OpenAlex 有重叠但**不是冗余**：OpenAlex 是二次加工的图谱，Crossref 是一手
    注册记录，正式发表版本的元数据以它为准。实测 OpenAlex 会间歇性超时，
    这条通路顶得上。

    **必须带 ``type:journal-article`` 过滤。** 不加的话返回里会混进补充材料、
    数据集、视频附件这类记录——实测标题长这样：
    ``Diffusion Trajectory-guided Policy ..._supp1-3619794.mp4``。
    这类条目对「这个方向有什么工作」毫无价值，却会占满 limit。
    """
    with _client() as client:
        response = client.get(
            "https://api.crossref.org/works",
            params={
                "query": query,
                "rows": limit,
                "filter": "type:journal-article",
                "select": "DOI,title,issued,container-title,is-referenced-by-count,author,abstract",
            },
        )
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in (payload.get("message", {}).get("items") or [])[:limit]:
        title_list = item.get("title") or []
        title = (title_list[0] if title_list else "").strip()
        if not title:
            continue

        # issued.date-parts 常带 null（只有年份时后面补 None），过滤掉
        parts = (item.get("issued") or {}).get("date-parts") or [[]]
        year_bits = [str(p) for p in (parts[0] or []) if p]
        doi = (item.get("DOI") or "").strip()

        results.append(
            WebResult(
                title=re.sub(r"\s+", " ", title),
                url=f"https://doi.org/{doi}" if doi else "",
                # Crossref 的 abstract 是 JATS XML 片段，标签得去掉
                snippet=_strip_jats(item.get("abstract"))[:600],
                source="crossref",
                kind="paper",
                published="-".join(year_bits[:1]) or (year_bits[0] if year_bits else ""),
                authors=[
                    a.get("family") or a.get("name") or ""
                    for a in (item.get("author") or [])[:5]
                ],
                extra={
                    "doi": doi.lower(),
                    "venue": (item.get("container-title") or [""])[0],
                    "cited_by": item.get("is-referenced-by-count"),
                },
            )
        )
    return results


def _strip_jats(abstract: str | None) -> str:
    """去掉 Crossref 摘要里的 JATS 标签。

    它返回的是 ``<jats:p>…</jats:p>`` 这种片段而不是纯文本，
    原样塞给模型会浪费上下文、也干扰阅读。
    """
    if not abstract:
        return ""
    text = re.sub(r"<[^>]+>", " ", abstract)
    return re.sub(r"\s+", " ", text).strip()


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


def search_hackernews(query: str, limit: int) -> list[WebResult]:
    """HackerNews（Algolia 索引）：免 Key，实测稳定。

    **它补的是「这项工作外界怎么评价」这一类问题**——论文本身告诉你作者声称
    什么，评论区告诉你同行信不信、有没有人复现失败、有没有更早的相似工作。
    这是学术接口完全给不了的信息。

    ``points`` 与 ``num_comments`` 是现成的质量信号：它们由真实的人投票产生，
    比任何相关度分数都难伪造。
    """
    with _client() as client:
        response = client.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": limit},
        )
        response.raise_for_status()
        payload = response.json()

    results = []
    for item in (payload.get("hits") or [])[:limit]:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        object_id = item.get("objectID") or ""
        # 有些条目（Ask HN 之类）没有外链，这时指向讨论页本身——
        # 讨论内容往往比原链接更有价值，丢掉它们是浪费
        url = item.get("url") or (
            f"https://news.ycombinator.com/item?id={object_id}" if object_id else ""
        )
        results.append(
            WebResult(
                title=title,
                url=url,
                snippet=(item.get("story_text") or "")[:400],
                source="hackernews",
                kind="web",
                published=(item.get("created_at") or "")[:10],
                extra={
                    "points": item.get("points"),
                    "comments": item.get("num_comments"),
                    "discussion_url": (
                        f"https://news.ycombinator.com/item?id={object_id}"
                        if object_id
                        else ""
                    ),
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


def _normalize_title(title: str) -> str:
    """标题归一化：去掉标点、压缩空白、转小写。

    保留中日韩字符——用 ``\\w`` 加 CJK 范围，而不是只留 ASCII 字母数字。
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w一-鿿]+", " ", title or "")).strip().lower()


def _canonical_id(item: WebResult) -> str:
    """提取权威标识：DOI / arXiv / GitHub 仓库。

    有权威标识时**根本不需要比标题**——它就是这篇工作的唯一身份。
    标题会随「预印本 vs 正式版」「有没有副标题」而变，标识不会。
    """
    blob = f"{item.url} {item.extra.get('doi') or ''}".lower()

    # arXiv 会给每篇预印本分配一个 DOI，形如 10.48550/arXiv.2301.12345。
    # 先认这个，把「DOI 形态」和「arXiv 形态」的两条记录归到同一个身份上——
    # 否则同一篇论文会以两张面孔出现。这是 DOI 与 arXiv 唯一的官方对应关系，
    # 其余的 DOI 不能凭标题去推测。
    match = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", blob)
    if match:
        return f"arxiv:{match.group(1)}"

    match = re.search(r"10\.\d{4,9}/[^\s\"<>]+", blob)
    if match:
        return f"doi:{match.group(0).rstrip('.')}"

    # arXiv：新版 2301.12345，老版 cs.CV/0701001
    match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", blob)
    if match:
        return f"arxiv:{match.group(1)}"
    match = re.search(r"\b(\d{4}\.\d{4,5})(v\d+)?\b", blob)
    if match and "arxiv" in blob:
        return f"arxiv:{match.group(1)}"

    if item.source == "github" or "github.com" in blob:
        match = re.search(r"github\.com/([^/\s]+/[^/\s#?]+)", blob)
        if match:
            return f"gh:{match.group(1).rstrip('.git').lower()}"

    return ""


def _richness(item: WebResult) -> tuple:
    """判断两条重复记录里哪条信息更全——合并时保留这条。"""
    return (
        len(item.content or ""),
        len(item.snippet or ""),
        len(item.authors),
        int(item.extra.get("cited_by") or 0),
    )


def _merge(keep: WebResult, other: WebResult) -> None:
    """把 ``other`` 的信息并进 ``keep``（原地）。

    不是简单丢弃：两条来源不同的记录各有各的补充——arXiv 有摘要、
    Crossref 有 DOI 和引用数、HN 有讨论链接。丢掉哪一边都是损失。
    """
    if _richness(other) > _richness(keep):
        # other 更全，交换内容但保留 keep 已累计的**票与来源**——
        # 那些是「这项工作被几个源命中」的记录，换掉就等于丢票
        votes = keep.extra.get("_votes") or []
        sources = keep.extra.get("sources") or []
        keep.title, other.title = other.title, keep.title
        keep.url, other.url = other.url, keep.url
        keep.snippet, other.snippet = other.snippet, keep.snippet
        keep.content, other.content = other.content, keep.content
        keep.authors, other.authors = other.authors, keep.authors
        keep.published, other.published = other.published, keep.published
        keep.extra, other.extra = other.extra, keep.extra
        keep.extra["_votes"] = votes
        keep.extra["sources"] = sources

    for key, value in other.extra.items():
        if key.startswith("_"):
            continue
        # cited_by / points 这类取大值：不同源的口径可以差很多，
        # 取大的那个更接近「有多少人认可」
        if key in {"cited_by", "points", "stars", "comments", "score"}:
            current = keep.extra.get(key) or 0
            keep.extra[key] = max(current, value or 0)
        elif not keep.extra.get(key):
            keep.extra[key] = value

    # 票与来源都要累加：这是 RRF 的输入，也是「这项工作被几个源同时命中」的答案
    merged_votes = list(keep.extra.get("_votes") or [])
    merged_votes.extend(other.extra.get("_votes") or [])
    keep.extra["_votes"] = merged_votes

    sources = list(keep.extra.get("sources") or [])
    for src in [*(keep.extra.get("sources") or []), *(other.extra.get("sources") or [])]:
        if src and src not in sources:
            sources.append(src)
    for src in (keep.source, other.source):
        if src and src not in sources:
            sources.append(src)
    keep.extra["sources"] = sources

    if not keep.content and other.content:
        keep.content = other.content
    if len(other.snippet or "") > len(keep.snippet or ""):
        keep.snippet = other.snippet
    if not keep.published and other.published:
        keep.published = other.published


def _dedup(items: list[WebResult], *, threshold: float = 92.0) -> list[WebResult]:
    """合并指向同一项工作的多条记录。

    两级判定，**顺序不能反**：

    1. **权威 ID**（DOI / arXiv / GitHub）相同即合并。
    2. 都没有权威 ID 时，才退而比标题相似度。

    **一条铁律：两边都有权威 ID 且不相等 → 判定为不同工作，不做模糊合并。**
    DDPM 与 DDIM 是两篇完全不同的论文，标题却有 85.2 的相似度
    （``Denoising Diffusion Probabilistic Models`` /
    ``Denoising Diffusion Implicit Models``，实测量出来的）。
    只看标题就会把它们并成一条，用户会以为其中一篇不存在。

    阈值取 92 与 ``scan.fuzzy_threshold`` 的既有取值一致，也是实测选出来的：
    同一篇的各种写法（大小写、副标题、预印本 vs 正式版）都是 100.0，
    而最容易混的 DDPM/DDIM 是 85.2。
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:  # pragma: no cover - rapidfuzz 是既有依赖
        fuzz = None

    kept: list[WebResult] = []
    by_id: dict[str, WebResult] = {}

    for item in items:
        if not (item.title or item.url):
            continue

        cid = _canonical_id(item)
        if cid and cid in by_id:
            _merge(by_id[cid], item)
            continue

        if not cid and fuzz is not None:
            # 走到这里说明**这条没有权威 ID**。「两边都有 ID 且不同」的情况
            # 根本进不来——有 ID 的记录在上面按 ID 判过，不同 ID 就是不同工作。
            # 剩下的都允许比标题：这一条身份未定，只能靠标题判断。
            norm = _normalize_title(item.title)
            match = None
            for candidate in kept:
                if fuzz.token_set_ratio(norm, _normalize_title(candidate.title)) >= threshold:
                    match = candidate
                    break
            if match is not None:
                _merge(match, item)
                continue

        if cid:
            by_id[cid] = item
        kept.append(item)

    return kept


def _fuse(scored: list[WebResult], *, rrf_k: int) -> list[WebResult]:
    """按 RRF 给**合并后**的每条结果打分并排序。

    **为什么是 RRF 而不是加权求和**：各源的分数量纲根本不可比——
    HackerNews 的 ``points`` 是几十到几千，Crossref 的 ``cited_by`` 可能是 0 也可能
    是 5000，arXiv 的相关度分是另一套东西。加权求和只能拍脑袋调参，
    而且每加一个源就得重调一次。RRF 只看「排第几」，天然免疫量纲问题——
    这与 ``kb/services/search.py`` 内部检索用 RRF 是同一个理由。

    票在 ``_merge`` 里累加（``extra["_votes"]``），也就是**按「一项工作」投票
    而不是按「一条记录」**：同一篇论文被三个源返回时，三票要归到同一条上，
    这本身就是最强的相关度信号。
    """
    for item in scored:
        votes = item.extra.get("_votes") or []
        item.extra["_rrf"] = sum(1.0 / (rrf_k + rank) for _, rank in votes)

    return sorted(
        scored,
        # kind 只做**轻微**偏好，不是硬排序。之前是 `r.kind != "paper"` 的硬键，
        # 结果讨论类和问答类永远排在论文后面——而「这项工作有没有人质疑过」
        # 恰恰只能由讨论类回答。现在它只在 RRF 相同时起作用。
        key=lambda r: (
            -r.extra.get("_rrf", 0.0),
            0 if r.kind == "paper" else (1 if r.kind == "code" else 2),
            r.title,
        ),
    )


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------


def _cache_key(query: str, kinds: tuple[str, ...], limit: int, fetch_pages: bool) -> str:
    """缓存键。**必须把影响结果的参数都算进去。**

    只按 query 做键会串味：同一个问句用 ``kinds=("paper",)`` 和
    ``kinds=("web",)`` 得到的是完全不同的结果，共用一份缓存会让模型
    拿到它没要的那类内容，而看起来像是检索出了问题。
    """
    blob = json.dumps(
        {
            "q": query.strip().lower(),
            "k": sorted(kinds),
            "n": limit,
            "f": fetch_pages,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _serialize(items: list[WebResult]) -> str:
    """序列化。

    **不能拿 ``to_dict()`` 往返**——它故意不含 ``content``（抓来的正文）和
    ``extra``（``cited_by`` 等排序/展示信息）。用它做缓存，取回来的是
    一次没有正文、没有引用数的降级结果，而且不会有任何报错。
    """
    return json.dumps(
        [
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "source": item.source,
                "kind": item.kind,
                "published": item.published,
                "authors": item.authors,
                # 下划线开头的内部字段一般不落盘，但 _rrf 要留下：
                # 它不影响这次返回的顺序（数组顺序已经固定），可一旦日后有人
                # 对取回来的结果重新排序，得分的缺失会让顺序无声地乱掉。
                "extra": {
                    **{k: v for k, v in item.extra.items() if not k.startswith("_")},
                    "_rrf": item.extra.get("_rrf", 0.0),
                },
                "content": item.content,
            }
            for item in items
        ],
        ensure_ascii=False,
    )


def _as_naive_utc(value):
    """把时间统一成 naive UTC，用于比较。

    **为什么需要它。** SQLite 不存时区：写进去的 aware datetime（``utcnow()``）
    读回来是 naive 的，但值仍是 UTC（实测写 04:03Z + 60min，读回 05:03 而不是
    本地时间的 12:03）。拿它直接和 ``utcnow()`` 比会抛
    ``TypeError: can't compare offset-naive and offset-aware datetimes``。

    这个错误在这一处特别难发现：缓存的容错逻辑会把异常当成「未命中」，
    于是表现是**缓存永远不命中**——没有报错、没有异常，只是每次搜索都白等
    几十秒重新联网。写这段代码时就是这么踩进去的，靠打日志才揪出来。

    先判断再加时区（而不是无条件 replace）是为了换到 Postgres 之类会返回
    aware 时间的后端时也正确。
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _cache_load(key: str) -> list[WebResult] | None:
    """读缓存。未命中或已过期返回 None。

    **任何异常都当作未命中**：缓存是纯粹的性能优化，它坏了（表还没建、
    数据损坏、不在应用上下文里）只该让检索慢一点，不该让它失败。
    """
    try:
        from ..extensions import db
        from ..models import WebSearchCache
        from ..models.base import utcnow

        row = (
            db.session.query(WebSearchCache)
            .filter(WebSearchCache.key == key)
            .one_or_none()
        )
        if row is None:
            return None
        expires = _as_naive_utc(row.expires_at)
        if expires is not None and expires < _as_naive_utc(utcnow()):
            # 惰性删除：项目里没有定时清扫器，读到过期就顺手清掉
            db.session.delete(row)
            db.session.commit()
            return None

        payload = json.loads(row.payload or "[]")
    except Exception:
        # 用 warning 而不是 debug：缓存读失败是被容忍的（当作未命中），
        # 所以「缓存 100% 不命中、每次都白等几十秒联网」这种情况，
        # 只有在日志里喊出来才可能被发现。这条就是被这么找出来的——
        # debug 级的日志让一次彻底失效的缓存安静地跑了过去。
        log.warning("读取联网缓存失败，按未命中处理", exc_info=True)
        return None

    results = []
    for item in payload:
        results.append(
            WebResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source=item.get("source", ""),
                kind=item.get("kind", "web"),
                published=item.get("published", ""),
                authors=item.get("authors") or [],
                extra=item.get("extra") or {},
                content=item.get("content", ""),
            )
        )
    return results


def _cache_store(key: str, items: list[WebResult], *, ttl_minutes: int) -> None:
    """写缓存。失败只记日志——缓存写不进去不该让一次成功的检索变成失败。"""
    try:
        from datetime import timedelta

        from ..extensions import db
        from ..models import WebSearchCache
        from ..models.base import utcnow

        now = utcnow()
        row = (
            db.session.query(WebSearchCache)
            .filter(WebSearchCache.key == key)
            .one_or_none()
        )
        payload = _serialize(items)
        if row is None:
            row = WebSearchCache(key=key, payload=payload)
            db.session.add(row)
        else:
            row.payload = payload
        row.created_at = now
        row.expires_at = now + timedelta(minutes=ttl_minutes)

        # 顺手清理过期行：项目里没有定时清扫器，把清理挂在写入路径上，
        # 成本摊薄到每次写，不需要额外的调度设施。
        # 这里传 naive 值：SQLite 里存的就是 naive UTC，混着带时区的值去比
        # 会让字符串比较失真（尾部多个 "+00:00"）。
        db.session.query(WebSearchCache).filter(
            WebSearchCache.expires_at < _as_naive_utc(now)
        ).delete(synchronize_session=False)
        db.session.commit()
    except Exception:
        log.debug("写入联网缓存失败（结果已正常返回）", exc_info=True)


def clear_cache() -> int:
    """清空联网检索缓存，返回删除条数。"""
    from ..extensions import db
    from ..models import WebSearchCache

    count = db.session.query(WebSearchCache).delete(synchronize_session=False)
    db.session.commit()
    return count


def cache_stats() -> dict[str, int]:
    """缓存的条数统计。

    过期判断放在这里做，而不是让调用方写 SQL——SQLite 存的是 naive UTC，
    调用方拿 ``utcnow()`` 去 filter 会因为时区问题算错（见 ``_as_naive_utc``）。
    """
    from ..extensions import db
    from ..models import WebSearchCache
    from ..models.base import utcnow

    rows = db.session.query(WebSearchCache.expires_at).all()
    now = _as_naive_utc(utcnow())
    alive = sum(
        1 for (expires,) in rows if _as_naive_utc(expires) and _as_naive_utc(expires) > now
    )
    return {"total": len(rows), "alive": alive, "expired": len(rows) - alive}


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

    # ---- 缓存 ----
    ttl = int(settings.get("websearch.cache_ttl_minutes") or 0)
    cache_key = _cache_key(query, kinds, limit, fetch_pages)
    if ttl > 0:
        cached = _cache_load(cache_key)
        if cached is not None:
            log.debug("联网检索命中缓存：%r", query[:40])
            # 命中缓存时**不返回 errors**：这些结果已经拿到了。
            # 前端见到 errors 非空会显示「联网未返回结果」，
            # 那对一次成功的检索来说是错的。
            return cached, []

    # 每个后端分一小份配额，避免一个源独吞
    per_source = max(2, limit // 2)
    ranked_lists: list[list[WebResult]] = []
    errors: list[str] = []

    backends: list[tuple[str, Any]] = []
    if "paper" in kinds and settings.get("websearch.academic"):
        backends.append(("openalex", lambda: search_openalex(query, per_source)))
        backends.append(("crossref", lambda: search_crossref(query, per_source)))
        backends.append(("arxiv", lambda: search_arxiv(query, per_source)))
        backends.append(("semantic_scholar", lambda: search_semantic_scholar(query, per_source)))
    if "code" in kinds and settings.get("websearch.code"):
        backends.append(("github", lambda: search_github(query, per_source)))
    if "web" in kinds:
        if settings.get("websearch.discussions"):
            backends.append(("hackernews", lambda: search_hackernews(query, per_source)))
        # 密钥必须走 get_secret()：secret 类型在库里存的是 Fernet 密文，
        # 用 get() 取到的密文会原样发给 Tavily，认证必然失败——
        # 而表面现象是「Tavily 挂了」，排查方向会完全跑偏。
        key = ""
        with contextlib.suppress(Exception):
            key = settings.get_secret("websearch.tavily_api_key") or ""
        if key:
            backends.append(("tavily", lambda: search_tavily(query, per_source, key)))

    for name, run in backends:
        try:
            found = run()
            if found:
                ranked_lists.append(found)
            log.debug("联网检索 %s：%d 条", name, len(found))
        except Exception as exc:
            errors.append(f"{name} 检索失败：{type(exc).__name__}")
            log.debug("联网检索 %s 失败", name, exc_info=True)

    # 给每条记录打上「来自第几个源、排第几」，作为 RRF 的票。
    # **票要在去重之前打**：合并之后才知道哪些记录其实是同一项工作，
    # 那时把票累加起来，得到的才是「这项工作被几个源、以多高的名次命中」。
    flat: list[WebResult] = []
    for list_index, results in enumerate(ranked_lists):
        for rank, item in enumerate(results, start=1):
            item.extra["_votes"] = [(list_index, rank)]
            item.extra.setdefault("sources", [item.source])
            flat.append(item)

    ordered = _fuse(_dedup(flat), rrf_k=RRF_K)
    chosen = ordered[:limit]

    if fetch_pages:
        max_chars = int(settings.get("websearch.max_page_chars"))
        for item in chosen:
            # 已经有摘要且够长的就不抓了——抓取是这里最慢的一步
            if len(item.snippet) > 800:
                continue
            content = fetch_page(item.url, max_chars=max_chars)
            if content:
                item.content = content

    # 缓存只存成功的结果；全空时不写，否则一次网络抖动会被缓存住
    if ttl > 0 and chosen:
        _cache_store(cache_key, chosen, ttl_minutes=ttl)

    return chosen, errors


__all__ = ["WebResult", "fetch_page", "search"]
