"""混合检索：全文 + 向量，用 RRF 融合。

**为什么需要融合而不是直接加权求和。** BM25 的分数范围随语料变化（同一查询
在不同库上可能是 3 分也可能是 30 分），余弦相似度固定在 -1..1。
两者量纲不同，加权求和需要为每个库重新调参，换个库就失效。

RRF（Reciprocal Rank Fusion）只用**名次**不用分数：``Σ 1/(k + rank)``。
它天然免疫量纲问题，也不怕某一侧的绝对分数漂移。代价是丢掉了「差多远」
这个信息——但对「把最相关的几条排到前面」这个目标来说，名次比分数可靠。

**中文的两条路径。** trigram 分词器对 3 字以上的片段才建索引，而「模型」
「点积」这类两字词在中文技术语料里极常见。所以短查询走 LIKE 扫描——
实测 2 万分块下约 2.6ms，完全可用。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from ..extensions import db
from ..models import Chunk, Note, Paper
from . import fts as fts_service

log = logging.getLogger(__name__)

# RRF 的平滑常数。60 是原论文的经验值：它决定「名次差异」的衰减速度，
# 太大则前排优势被抹平，太小则单一来源的偶然命中会压过另一来源的一致命中。
DEFAULT_RRF_K = 60


@dataclass
class SearchHit:
    """一条检索结果。带齐了生成引用所需的一切。"""

    chunk_id: str
    text: str
    score: float
    paper_id: str | None = None
    note_id: str | None = None
    paper_title: str | None = None
    note_title: str | None = None
    section_path: str | None = None
    page_from: int | None = None
    page_to: int | None = None
    kind: str = "text"
    sources: list[str] = field(default_factory=list)  # fts_en / fts_cjk / vector / like
    snippet: str = ""

    @property
    def locator(self) -> str:
        """人类可读的定位串，直接用于引用标记。"""
        bits = []
        if self.section_path:
            bits.append(f"§{self.section_path}")
        if self.page_from:
            if self.page_to and self.page_to != self.page_from:
                bits.append(f"p.{self.page_from}-{self.page_to}")
            else:
                bits.append(f"p.{self.page_from}")
        return " ".join(bits)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "note_id": self.note_id,
            "paper_title": self.paper_title,
            "note_title": self.note_title,
            "section_path": self.section_path,
            "page_from": self.page_from,
            "page_to": self.page_to,
            "kind": self.kind,
            "score": round(self.score, 6),
            "sources": self.sources,
            "text": self.text,
            "snippet": self.snippet or self.text[:300],
            "locator": self.locator,
        }


# --------------------------------------------------------------------------
# 片段高亮
# --------------------------------------------------------------------------


def _query_terms(query: str) -> list[str]:
    """从查询里取出用于高亮的词。

    英文按空白与标点切，中文按 2-gram/3-gram 切——中文没有空格，
    整句当词是匹配不上任何东西的。
    """
    terms: list[str] = []
    for token in re.split(r"[\s,，。;；:：!！?？()（）\[\]{}<>\"'`]+", query):
        token = token.strip()
        if not token:
            continue
        if any("一" <= ch <= "鿿" for ch in token):
            # 中文：连续片段本身，加上前 2、3 字的子串（提高命中率）
            terms.append(token)
            for size in (2, 3):
                if len(token) > size:
                    terms.append(token[:size])
        elif len(token) >= 2:
            terms.append(token)
    # 去重、长的优先（长词命中更有信息量）
    seen: set[str] = set()
    unique = []
    for term in sorted(terms, key=len, reverse=True):
        lowered = term.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(term)
    return unique


def make_snippet(text: str, query: str, *, width: int = 240) -> str:
    """截取包含查询词的片段，并标记出来。

    不用 FTS5 的 ``snippet()``：它只对建索引时用的分词器有效，
    而中文走的是 trigram 或 LIKE 路径，两边行为不一致。
    自己实现反而能保证中英文表现统一。
    """
    if not text:
        return ""

    terms = _query_terms(query)
    position = -1
    for term in terms:
        found = text.lower().find(term.lower())
        if found >= 0 and (position < 0 or found < position):
            position = found

    if position < 0:
        return text[:width] + ("…" if len(text) > width else "")

    start = max(0, position - width // 3)
    end = min(len(text), start + width)
    snippet = text[start:end]

    # 标记命中词（用特殊字符包裹，前端再渲染成 <mark>）。
    #
    # 两个要点：
    #   1. **不直接输出 HTML**。片段来自论文原文，而论文是用户上传的任意内容，
    #      在这里拼 HTML 等于开了一个 XSS 口子。用控制字符占位，由前端转换。
    #   2. **一次替换完成**。逐词循环替换的话，后一个词会匹配到前一个词
    #      已经加上的标记内部，出现 "⌈⌈基准⌉测⌉" 这样的嵌套标记——
    #      因为 3-gram 词表里「基准测试」「基准」「基准测」本来就是互相重叠的。
    #      合成一个正则交替式，长的排前面，就能保证每个位置只标记一次。
    words = [term for term in terms if len(term) >= 2]
    if words:
        pattern = "|".join(re.escape(term) for term in words)
        snippet = re.sub(f"({pattern})", "\u0002\\1\u0003", snippet, flags=re.I)

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


# --------------------------------------------------------------------------
# 各检索通道
# --------------------------------------------------------------------------


def _filter_clause(filters: dict) -> tuple[str, dict]:
    """构造附加的 WHERE 条件。返回 ``(SQL 片段, 参数字典)``。

    **安全性说明（这个片段会被拼进 SQL 文本，值得写清楚）：**
    返回的每一段都是本函数里写死的字面量，用户提供的值一律通过
    ``:name`` 占位符走参数绑定。也就是说，这个片段的内容只取决于
    「哪些筛选条件存在」，与条件的**取值**无关——取值进了 params。
    所以把它拼进查询是安全的，bandit 的 S608 在这两处是误报。
    """
    clauses: list[str] = []
    params: dict = {}

    if filters.get("paper_id"):
        clauses.append("AND chunks.paper_id = :f_paper")
        params["f_paper"] = filters["paper_id"]
    if filters.get("note_id"):
        clauses.append("AND chunks.note_id = :f_note")
        params["f_note"] = filters["note_id"]
    if filters.get("kind"):
        clauses.append("AND chunks.kind = :f_kind")
        params["f_kind"] = filters["kind"]
    if filters.get("paper_ids"):
        placeholders = ",".join(f":fp{i}" for i in range(len(filters["paper_ids"])))
        clauses.append(f"AND chunks.paper_id IN ({placeholders})")
        for i, value in enumerate(filters["paper_ids"]):
            params[f"fp{i}"] = value

    return " ".join(clauses), params


def _run_fts_match(
    table: str, match_expression: str, limit: int, filters: dict
) -> list[tuple[str, float]]:
    extra_where, extra_params = _filter_clause(filters)
    sql = fts_service.build_search_sql(table, limit=limit, extra_where=extra_where)
    params = {"query": match_expression, "limit": limit, **extra_params}
    try:
        with db.engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return [(row[0], float(row[8] or 0.0)) for row in rows]
    except Exception as exc:
        log.warning("FTS 查询失败（%s）：%s", table, exc)
        return []


def _fts_channel(table: str, query: str, limit: int, filters: dict) -> list[tuple[str, float]]:
    """查询单张 FTS 表。返回 ``[(chunk_id, score)]``。

    **按「精确 -> 宽松 -> 只查技术词 -> 中文拆元组」四步降级。**

    这是为「中文提问 + 英文语料」这个很常见的组合准备的。问
    「变分自编码器的 ELBO 是怎么推导出来的」时，AND 要求每个词都出现，
    而中文的表述架子（「是怎么推导出来的」）在英文论文里当然找不到，
    于是整条查询零结果——尽管 ELBO 就在语料里躺着。

    降级顺序不能反：先宽松会召回一堆只沾一个边的内容，把精确结果淹没。

    第 4 步是**不需要模型的兜底**。前三步对纯中文长句会全部落空：
    整句匹配不到，句子里又没有任何 ASCII 技术词可以退而求其次。
    此前这种情况唯一能救回来的是查询扩展那一步 LLM 调用——于是
    「LLM 不可用」（超预算、断网、没配 key）就等于「这类问句永远搜不到」，
    而界面上只会说「没找到相关内容」。拆成 3 元组之后，检索不再
    依赖任何模型调用。
    """
    # 1) 所有词都要出现（最精确）
    expression = fts_service.build_match_query(query, "AND")
    if expression:
        results = _run_fts_match(table, expression, limit, filters)
        if results:
            return results

    # 2) 出现任一即可（保召回）
    expression = fts_service.build_match_query(query, "OR")
    if expression:
        results = _run_fts_match(table, expression, limit, filters)
        if results:
            log.debug("[%s] AND 无结果，改用 OR 命中 %d 条", table, len(results))
            return results

    # 3) 只查 ASCII 技术词——中文提问命中英文论文最有效的一招
    technical = fts_service.extract_ascii_terms(query)
    if technical and technical.lower() != query.lower():
        expression = fts_service.build_match_query(technical, "OR")
        if expression:
            results = _run_fts_match(table, expression, limit, filters)
            if results:
                log.debug("[%s] 改用技术词 %r 命中 %d 条", table, technical, len(results))
                return results

    # 4) 中文长句拆成 3 元组。最后一道，只有前面全空才走这里——
    # 这就意味着「返回一点东西」严格优于「返回空」，不存在把精确结果淹掉的风险。
    grams = fts_service.cjk_ngrams(query)
    if len(grams) > 1:  # 只有一个元组时说明第 1/2 步已经试过它了
        expression = fts_service.build_match_query(" ".join(grams), "OR")
        if expression:
            results = _run_fts_match(table, expression, limit, filters)
            if results:
                log.debug(
                    "[%s] 整句无结果，拆成 %d 个中文元组命中 %d 条",
                    table, len(grams), len(results),
                )
                return results

    return []


def _expanded_terms(query: str, filters: dict) -> str:
    """取查询的英文扩展词。未启用或失败时返回空串。"""
    try:
        from flask import current_app

        settings = current_app.extensions["kb_settings"]
        if not settings.get("retrieval.query_expansion"):
            return ""
        from .llm import is_configured

        if not is_configured():
            return ""
    except Exception:
        # 不在应用上下文里（脚本调用）时静默跳过，不影响检索本身
        return ""

    try:
        from .query_expand import expand_query

        terms = expand_query(query)
        return " ".join(terms) if terms else ""
    except Exception as exc:
        log.debug("查询扩展失败：%s", exc)
        return ""


def _like_channel(query: str, limit: int, filters: dict) -> list[tuple[str, float]]:
    """短中文查询的兜底通道：LIKE 扫描。

    trigram 对 2 字词无效，而 2 字词在中文里太常见，不能不管。
    给一个较低的基准分，让它在与 FTS 结果融合时排在后面——
    它是「宁可多召回」的通道，不是主力。
    """
    extra_where, extra_params = _filter_clause(filters)
    # 过滤掉 LIKE 的通配符，否则用户输入 % 会匹配一切
    pattern = "%" + query.replace("%", "").replace("_", "").strip() + "%"

    # extra_where 由 _filter_clause 生成，只含字面量，取值走参数绑定
    sql = f"""
        SELECT chunks.id, chunks.text
          FROM chunks
         WHERE chunks.text LIKE :pattern
           {extra_where}
         ORDER BY length(chunks.text)
         LIMIT :limit
    """
    try:
        with db.engine.connect() as conn:
            rows = conn.execute(
                text(sql), {"pattern": pattern, "limit": limit, **extra_params}
            ).fetchall()
    except Exception as exc:
        log.warning("LIKE 兜底查询失败：%s", exc)
        return []

    # 用出现次数当粗略相关性：命中越多（相对长度越短）越相关
    scored = []
    for chunk_id, content in rows:
        occurrences = content.lower().count(query.lower()) if content else 0
        density = occurrences / max(1, len(content) / 1000)
        scored.append((chunk_id, density))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def _vector_channel(query: str, limit: int, filters: dict) -> list[tuple[str, float]]:
    """向量检索通道。未配置或失败时返回空列表——检索继续走全文通道。"""
    from . import embedding as embedding_service

    try:
        provider = embedding_service.get_provider()
        if provider is None:
            return []
        model = embedding_service.active_model()
        if model is None:
            return []

        vectors = provider.embed([query])
        if not vectors:
            return []
        hits = embedding_service.search_vectors(model, vectors[0], limit * 3)

        # 向量表里可能有已删论文/笔记留下的孤儿，这里过滤掉
        allowed = {hit[0] for hit in hits}
        if not allowed:
            return []
        placeholders = ",".join(f":c{i}" for i in range(len(allowed)))
        params = {f"c{i}": value for i, value in enumerate(allowed)}
        # placeholders 是 :c0,:c1,… 形式的占位符，不是值
        sql = f"SELECT id FROM chunks WHERE id IN ({placeholders})"
        extra_where, extra_params = _filter_clause(filters)
        if extra_where:
            sql += f" AND 1=1 {extra_where}"
            params.update(extra_params)

        with db.engine.connect() as conn:
            valid = {row[0] for row in conn.execute(text(sql), params)}

        return [(cid, score) for cid, score in hits if cid in valid][:limit]
    except Exception as exc:
        log.warning("向量检索失败，本次仅用全文检索：%s", exc)
        return []


# --------------------------------------------------------------------------
# RRF 融合
# --------------------------------------------------------------------------


# 标题里的功能词，算缩写时要跳过——留着它们缩写会变成 "totm" 这种噪声
_TITLE_STOPWORDS = frozenset({
    "for", "the", "a", "an", "of", "on", "in", "with", "and", "to", "via",
    "using", "toward", "towards", "from", "by", "at", "as", "is", "are",
    "based", "through", "without", "into", "over", "under", "new",
})


def _title_channel(query: str, limit: int, filters: dict) -> list[tuple[str, float]]:
    """标题 / 缩写精确命中通道。

    用户提问经常直接用论文简称（「DDPM」「UniAD」），但**这些简称在引用它的
    其它论文正文里出现得同样密集**，全文检索排不出高低。实测：问
    「DDPM 论文的核心贡献是什么？」，FTS 的前五条全是别的论文
    （Improved DDPM、DDIM、DiT），原论文反而不在其中。

    这条通道不看正文，直接拿查询去比标题：标题词有多少出现在查询里，
    以及标题的首字母缩写是否就是查询中的某个词。命中即强召回。

    缩写那一条尤其重要——「DDPM」这个词在正文里到处都是，
    只有 "Denoising Diffusion Probabilistic Models" 的缩写能唯一确定它。
    """
    scored = _title_scores(query)
    if not scored:
        return []

    top = scored[:5]
    top_ids = [item["paper_id"] for item in top]

    # 每篇取靠前的几块（摘要/引言最能代表这篇论文在讲什么）
    extra_where, extra_params = _filter_clause(filters)
    params: dict[str, Any] = {}
    placeholders = ",".join(f":p{i}" for i in range(len(top_ids)))
    params.update({f"p{i}": pid for i, pid in enumerate(top_ids)})
    sql = f"SELECT id, paper_id FROM chunks WHERE paper_id IN ({placeholders})"
    if extra_where:
        sql += f" AND 1=1 {extra_where}"
        params.update(extra_params)

    with db.engine.connect() as conn:
        rows = list(conn.execute(text(sql), params))

    if not rows:
        return []

    strength = {item["paper_id"]: item["score"] for item in scored}
    by_paper: dict[str, list[str]] = {}
    for chunk_id, paper_id in rows:
        by_paper.setdefault(paper_id, []).append(chunk_id)

    # 按论文得分排序输出；论文内部按 chunk id 稳定排序
    # （不按 ord 取前几块是因为这里只拿到了 id，再查一次 ord 不划算，
    #  而同一篇论文的块得分本来就相同，顺序不影响它在融合里的名次）
    ordered: list[tuple[str, float]] = []
    for item in top:
        paper_id = item["paper_id"]
        if paper_id not in by_paper:
            continue
        for chunk_id in sorted(by_paper[paper_id])[:3]:
            ordered.append((chunk_id, strength[paper_id]))
    return ordered[:limit]


def _title_scores(query: str) -> list[dict[str, Any]]:
    """论文标题与查询的匹配强度。通道与提权共用这一份判断。

    **必须共用**：如果通道按一套规则召回、提权按另一套规则筛选，
    迟早出现「提权了但通道没召回」或反过来的错位，而且极难排查。
    """
    from ..models import Paper

    query = (query or "").lower()
    if not query:
        return []
    # 查询里的词（含中文串），用于判断标题词是否出现
    query_tokens = set(re.findall(r"[a-z0-9][a-z0-9._-]{1,}", query))

    rows = (
        db.session.query(Paper)
        .filter(Paper.deleted_at.is_(None), Paper.title.isnot(None))
        .all()
    )
    if not rows:
        return []

    # ---- IDF：标题词的**稀有度**才是证据强度 ----
    #
    # 原先按「命中了几个标题词」算分，结果 "autonomous"、"driving"
    # 这类词在几十个标题里都有，命中的论文一拥而上——问 UniAD
    # 会把 GenAD 顶到第一（它的标题也含这两个词）。
    #
    # 换成 IDF 之后不需要人工维护停用词表：语料自己会说话，
    # 「planning-oriented」只出现在一篇标题里，权重自然远高于
    # 出现在几十篇里的「autonomous」。语料换了（比如从自动驾驶
    # 换成生物信息），这套权重会自己适配。
    # 标题词列表**必须保持原顺序**：缩写靠首字母按顺序拼出来，
    # 转成 set 再 sorted 会得到 "ddmp" 而不是 "ddpm"，缩写通道直接失效。
    title_words: list[list[str]] = []
    document_frequency: dict[str, int] = {}
    for paper in rows:
        words = [
            w for w in re.findall(r"[a-z0-9][a-z0-9-]{1,}", (paper.title or "").lower())
            if len(w) >= 3
        ]
        title_words.append(words)
        for word in set(words):
            document_frequency[word] = document_frequency.get(word, 0) + 1

    total = len(rows)

    def idf(word: str) -> float:
        return math.log((total + 1) / (document_frequency.get(word, 0) + 1))

    # 一个词都没在标题里出现时的权重（罕见词的天花板）
    acronym_value = math.log(total + 1)

    scored: list[tuple[float, str]] = []
    for paper, words in zip(rows, title_words, strict=True):
        if not words:
            continue

        # 缩写：跳过功能词后的首字母（按标题原顺序）。
        # 命中缩写几乎可以确定指的就是这篇，所以给一个独立的高分——
        # 「DDPM」在正文里到处都是，只有标题首字母能唯一确定它。
        initials = "".join(w[0] for w in words if w not in _TITLE_STOPWORDS)
        acronym_hit = len(initials) >= 2 and initials in query_tokens

        hits = [w for w in set(words) if w in query_tokens]
        if not hits and not acronym_hit:
            continue

        score = sum(idf(w) for w in hits)
        if acronym_hit:
            score += acronym_value

        # 命中太少又没有缩写支撑的，当噪声丢掉（避免「标题里有个词就算命中」）
        if not acronym_hit and len(hits) < 2:
            continue
        scored.append(
            {
                "paper_id": paper.id,
                "score": score,
                "acronym_hit": acronym_hit,
                "coverage": len(hits) / max(1, len(set(words))),
                # 命中的标题词里**最罕见**那个的 IDF。这是判断「有没有点名
                # 一篇论文」最可靠的一条——比总分和覆盖率都稳。
                "max_idf": max((idf(w) for w in hits), default=0.0),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


# 判定「强命中」的门槛：命中词里最罕见那个的 IDF。
#
# 试过另外两个判据，都不行：
#   * **IDF 总分**：长标题词多，凑出来的总量自然大。实测问「UniAD 是什么方法？",
#     UniAD 标题 3 个词全中得 5.4 分，而 ORION 那篇长标题只中 40% 却得 5.8 分，
#     被提权的是 ORION。
#   * **覆盖率**：能挡掉 ORION，但会误杀 VGGT——它的标题
#     "VGGT: Visual Geometry Grounded Transformer" 带副标题，
#     用户只可能说出 "VGGT" 一个词，覆盖率永远上不去。
#
# 「最罕见命中词」两个都挡得住：UniAD 的 planning-oriented、VGGT 的 vggt
# 各自只出现在一篇标题里（IDF≈3.8）；而 ORION 命中的 end-to-end、autonomous
# 和概念问题命中的 diffusion、models 都是几十篇标题里的常客（IDF<2.5）。
#
# 语义上也说得通：**一个词越罕见，它出现在查询里就越是有意为之**。
_STRONG_TITLE_MAX_IDF = 3.0


def _strong_title_matches(query: str) -> dict[str, float]:
    """查询是否**点名**了某篇论文，返回 ``{paper_id: 匹配强度}``。

    两条判据：
      * 命中标题首字母缩写（「DDPM」）；
      * 命中了一个**极罕见**的标题词（全语料只出现在少数标题里）。

    后者比缩写覆盖面更广：论文的标题常常含有自己独有的术语，
    用户说出那个词，基本就是在指这篇论文。
    """
    return {
        item["paper_id"]: item["max_idf"]
        for item in _title_scores(query)
        if item["acronym_hit"] or item["max_idf"] >= _STRONG_TITLE_MAX_IDF
    }


def _promote_papers(
    fused: list[tuple[str, float, list[str]]], matches: dict[str, float]
) -> list[tuple[str, float, list[str]]]:
    """把点名的论文的片段提到最前，其余保持原顺序。

    只调整顺序，不改分数——分数是各通道融合出来的事实，
    为了提权去篡改它会让「为什么这条排第一」变得无法解释。

    提上来的论文之间**按匹配强度排序**：同时点名多篇时，
    匹配得更完整的那篇应该排在前面。
    """
    if not fused or not matches:
        return fused

    chunk_ids = [item[0] for item in fused]
    placeholders = ",".join(f":c{i}" for i in range(len(chunk_ids)))
    params: dict[str, Any] = {f"c{i}": cid for i, cid in enumerate(chunk_ids)}
    id_list = ",".join(f":p{i}" for i in range(len(matches)))
    params.update({f"p{i}": pid for i, pid in enumerate(matches)})

    with db.engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT id, paper_id FROM chunks "
                f"WHERE id IN ({placeholders}) AND paper_id IN ({id_list})"
            ),
            params,
        )
        paper_of = {row[0]: row[1] for row in rows}

    promoted = [item for item in fused if item[0] in paper_of]
    promoted.sort(key=lambda item: matches.get(paper_of[item[0]], 0.0), reverse=True)
    rest = [item for item in fused if item[0] not in paper_of]
    return promoted + rest


def _discrimination(hits: list[tuple[str, float]]) -> float:
    """通道的区分度：头部得分相对整体分布的分离程度，0~1。

    **为什么需要这个。** 语料是英文论文，而当前的本地嵌入模型是
    bge-small-zh（中文）。它在英文文本上给出的相似度几乎全挤在 0.49 附近——
    也就是说这个通道对任何输入产生的排序都接近随机。把它和真正有信号的
    通道等权融合，等于往结果里掺噪声，而且是看不出来的那种：
    接口正常返回、分数看着也像模像样。

    区分度低就说明这一路说了等于没说，该降权而不是继续投票。
    """
    if len(hits) < 5:
        return 1.0  # 样本太少，不下结论
    scores = sorted((s for _, s in hits), reverse=True)
    top = scores[0]
    median = scores[len(scores) // 2]
    if top <= 0:
        return 0.0
    return max(0.0, (top - median) / abs(top))


def reciprocal_rank_fusion(
    channels: dict[str, list[tuple[str, float]]],
    k: int = DEFAULT_RRF_K,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float, list[str]]]:
    """把多路结果按名次融合。

    每条通道各自排名，然后 ``score = Σ weight / (k + rank)``。
    同时记录「这条结果是被哪几路召回的」——多路共同命中是比单路高分
    更可靠的信号，界面上也值得展示出来。

    **通道不该等权。** 精确匹配（标题、扩展出的技术术语）比语义通道
    可靠得多；而一个没有区分度的通道（见 ``_discrimination``）应该
    少投票甚至不投票。等权融合看起来「公平」，实际是让最弱的一路
    决定结果。
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    weights = weights or {}

    for name, hits in channels.items():
        weight = weights.get(name, 1.0)
        if weight <= 0:
            continue
        for rank, (chunk_id, _raw_score) in enumerate(hits):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank + 1)
            sources.setdefault(chunk_id, []).append(name)

    fused = [(cid, score, sources[cid]) for cid, score in scores.items()]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def _channel_weights(channels: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    """各通道的融合权重。

    权重按「这一路有多可信」给，而不是一律相等：

      * 标题/缩写精确命中是最强的信号——没有任何歧义；
      * 扩展出的英文术语是标准写法，字面匹配可靠；
      * LIKE 是短查询的兜底，精度天生低；
      * 向量通道**按实测区分度动态定权**（见下）。

    向量那一项是这里最重要的设计。实测当前配置（bge-small-zh 跑英文语料）
    的区分度只有 0.01~0.03——候选集内第一名和第六十名的相似度只差 1%，
    排序接近随机。而 RRF 只看名次，等于让这一路投噪声票。
    动态定权的好处是**换了好模型会自动恢复**：区分度上去了权重自然回来，
    不需要改代码，也不需要用户去理解为什么某个开关要关掉。
    """
    weights = {
        "title": 2.5,
        "fts_expanded": 1.2,
        "fts_en": 1.0,
        "fts_cjk": 1.0,
        "like": 0.6,
    }

    vector = channels.get("vector")
    if vector is not None:
        spread = _discrimination(vector)
        if spread < 0.05:
            log.info("向量通道区分度仅 %.3f，本次不参与融合（排名接近随机）", spread)
            weights["vector"] = 0.0
        else:
            # 0.05 → 0.5，0.15 及以上 → 1.0，中间线性过渡
            weights["vector"] = min(1.0, 0.5 + (spread - 0.05) * 5)
    return weights


def search(
    query: str,
    *,
    limit: int = 8,
    mode: str = "hybrid",
    filters: dict | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    group_by_paper: bool = False,
) -> list[SearchHit]:
    """混合检索。

    ``mode`` 可取 ``hybrid`` / ``fts`` / ``vector``，用于对比不同通道的效果，
    也方便在向量服务出问题时临时降级。
    """
    query = (query or "").strip()
    if not query:
        return []

    filters = dict(filters or {})
    # 多取一些候选，融合后再截断——每一路单独取 limit 条的话，
    # 融合几乎没有可排序的余地
    channel_limit = max(limit * 4, 20)

    channels: dict[str, list[tuple[str, float]]] = {}

    if mode in {"hybrid", "fts"}:
        channels["fts_en"] = _fts_channel(fts_service.FTS_EN, query, channel_limit, filters)

        if fts_service.min_trigram_len(query) >= 3:
            channels["fts_cjk"] = _fts_channel(
                fts_service.FTS_CJK, query, channel_limit, filters
            )
        elif fts_service.is_cjk(query):
            # 短中文查询：trigram 查不到，走 LIKE
            channels["like"] = _like_channel(query, channel_limit, filters)

        # 查询扩展：把中文提问翻成英文技术词，**作为额外一路**参与融合。
        #
        # 是「额外」而不是「替换」：中文原文走向量通道（语义），
        # 英文术语走全文通道（精确），各自的强项都用上。
        # 实测这一步对「中文问英文论文」的提升最明显——
        # 字面匹配的全文检索本来完全够不着那些查询。
        expanded = _expanded_terms(query, filters)
        if expanded:
            channels["fts_expanded"] = _fts_channel(
                fts_service.FTS_EN, expanded, channel_limit, filters
            )

        # 标题 / 缩写精确命中。**把扩展词也一起喂进去**：
        # 用户问「UniAD 是什么方法？」时，查询里没有任何标题词，
        # 但扩展那一步会产出 "planning-oriented autonomous driving"——
        # 只拿原查询去比标题就会漏掉。标题不在正文分块里，
        # 所以扩展词对正文检索有效、对标题匹配却完全用不上，这是个隐蔽的缺口。
        title_query = query if not expanded else f"{query} {' '.join(expanded)}"
        channels["title"] = _title_channel(title_query, channel_limit, filters)
        strong_papers = _strong_title_matches(title_query)

    if mode in {"hybrid", "vector"}:
        channels["vector"] = _vector_channel(query, channel_limit, filters)

    channels = {name: hits for name, hits in channels.items() if hits}
    if not channels:
        return []

    fused = reciprocal_rank_fusion(channels, k=rrf_k, weights=_channel_weights(channels))

    # 标题**强**命中的论文直接提到最前。
    #
    # 为什么不能只靠给 title 通道加权重：RRF 只看名次不看强度，而一篇论文
    # 在三条弱通道里各排第十（Σ≈0.050）会稳定压过在一条通道里排第一
    # （2.5/61≈0.041）。加大权重又会让「扩散模型的训练目标」这种概念问题
    # 被标题里恰好含 "diffusion models" 的论文劫持。
    #
    # 所以按**强度**分流：只有缩写命中、或罕见标题词命中足够多时才提权。
    # 用户点名了一篇论文（用缩写或标题），那就该返回那篇论文——
    # 这比任何语义相似度都确定。
    if strong_papers and mode != "vector":
        fused = _promote_papers(fused, strong_papers)
    if group_by_paper:
        fused = _group_by_paper(fused)

    # 取回文本与元数据
    top = fused[: limit * 2 if group_by_paper else limit]
    ids = [chunk_id for chunk_id, _, _ in top]

    rows = (
        db.session.query(Chunk, Paper, Note)
        .outerjoin(Paper, Chunk.paper_id == Paper.id)
        .outerjoin(Note, Chunk.note_id == Note.id)
        .filter(Chunk.id.in_(ids))
        .all()
    )
    index = {chunk.id: (chunk, paper, note) for chunk, paper, note in rows}

    hits: list[SearchHit] = []
    for chunk_id, score, sources in top:
        entry = index.get(chunk_id)
        if entry is None:
            continue
        chunk, paper, note = entry
        hits.append(
            SearchHit(
                chunk_id=chunk.id,
                text=chunk.text,
                score=score,
                paper_id=chunk.paper_id,
                note_id=chunk.note_id,
                paper_title=paper.title if paper else None,
                note_title=note.title if note else None,
                section_path=chunk.section_path,
                page_from=chunk.page_from,
                page_to=chunk.page_to,
                kind=chunk.kind,
                sources=sources,
                snippet=make_snippet(chunk.text, query),
            )
        )

    if group_by_paper:
        hits = _collapse_groups(hits, limit)

    return hits[:limit]


def _group_by_paper(fused: list[tuple[str, float, list[str]]]) -> list[tuple[str, float, list[str]]]:
    """把同一篇论文的连续命中聚到一起。

    做法是把同论文的最高分结果提到前面，其余同论文的紧随其后。
    这样既保留了「最相关的那篇排最前」，又不会让一篇论文的多个片段
    把别的论文挤出结果页。
    """
    if not fused:
        return fused

    chunk_ids = [item[0] for item in fused]
    rows = (
        db.session.query(Chunk.id, Chunk.paper_id, Chunk.note_id)
        .filter(Chunk.id.in_(chunk_ids))
        .all()
    )
    owner = {row[0]: (row[1] or row[2] or row[0]) for row in rows}

    buckets: dict[str, list[tuple[str, float, list[str]]]] = {}
    order: list[str] = []
    for item in fused:
        key = owner.get(item[0], item[0])
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)

    # 每个桶按桶内最高分排序，桶之间也按最高分排序
    order.sort(key=lambda key: -buckets[key][0][1])
    result: list[tuple[str, float, list[str]]] = []
    for key in order:
        result.extend(buckets[key])
    return result


def _collapse_groups(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    """按论文聚合：每篇论文最多保留 3 条，避免单篇刷屏。"""
    counts: dict[str, int] = {}
    kept: list[SearchHit] = []
    for hit in hits:
        key = hit.paper_id or hit.note_id or hit.chunk_id
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= 3:
            kept.append(hit)
        if len(kept) >= limit * 3:
            break
    return kept


def search_papers(
    query: str, *, limit: int = 20, filters: dict | None = None
) -> list[dict]:
    """按论文聚合的检索，供「找论文」这类查询用。

    与 ``search`` 的区别：这里返回的是一篇篇论文（附最相关的片段），
    而不是一条条片段。
    """
    hits = search(query, limit=limit * 4, filters=filters)
    grouped: dict[str, dict] = {}

    for hit in hits:
        key = hit.paper_id
        if key is None:
            continue
        if key not in grouped:
            grouped[key] = {
                "paper_id": key,
                "title": hit.paper_title,
                "best_score": hit.score,
                "snippets": [],
            }
        entry = grouped[key]
        if len(entry["snippets"]) < 3:
            entry["snippets"].append(
                {
                    "chunk_id": hit.chunk_id,
                    "text": hit.snippet,
                    "locator": hit.locator,
                    "page_from": hit.page_from,
                }
            )

    ordered = sorted(grouped.values(), key=lambda item: -item["best_score"])
    return ordered[:limit]


__all__ = [
    "DEFAULT_RRF_K",
    "SearchHit",
    "make_snippet",
    "reciprocal_rank_fusion",
    "search",
    "search_papers",
]
