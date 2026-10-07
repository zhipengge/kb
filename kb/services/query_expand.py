"""检索查询扩展：把中文提问变成能命中英文语料的检索词。

**为什么需要它。** 语料是英文论文，用户用中文提问，而全文检索是字面匹配——
「变分自编码器的优化目标是什么」里的每个字都不出现在英文论文里，
纯 FTS 零结果；向量检索能跨语言，但小模型的 top-1 准确率有限。

最直接有效的办法是**先把问题翻译成英文技术词再检索**。论文里的术语
是高度标准化的（"evidence lower bound"、"amortized inference"），
翻译过去之后是精确匹配，比语义向量的模糊匹配可靠得多。

成本很低：一次几百 token 的小调用，而且**按查询缓存**——
同一个问题反复问不会重复计费。

结果与原始查询**一起**参与检索（不是替换），两路用 RRF 融合：
中文原文走向量，英文术语走全文，各自的强项都用上。
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict

from . import budget

log = logging.getLogger(__name__)

# 查询扩展的缓存。用有序字典做简单的 LRU——检索查询重复率很高
# （反复追问同一篇论文时尤其明显），缓存能省掉绝大部分调用。
_CACHE: OrderedDict[str, list[str]] = OrderedDict()
_CACHE_MAX = 512

EXPAND_SYSTEM = """你把中文的论文检索问题转换成英文检索词。

要求：
- **每行一个**检索词或短语，输出 3-8 行，不要输出任何解释、编号或符号。
- 用这个领域的**标准术语**，而不是字面直译。例如「摊销变分推断」
  应该是 "amortized variational inference" 而不是 "share variational inference"。
- 覆盖问题里的每个关键概念，包括它们的常见英文同义说法。
- 如果问题里本来就有英文术语，原样保留并补上它的全称或缩写。

示例输出：

variational autoencoder
evidence lower bound
ELBO
amortized variational inference

只输出检索词本身。"""


def _cache_get(key: str) -> list[str] | None:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None


def _cache_put(key: str, value: list[str]) -> None:
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def _needs_expansion(query: str) -> bool:
    """判断是否值得做扩展。

    纯英文查询直接跳过——本来就能匹配上，多一次调用只是浪费。
    但含中文的查询即使夹着英文术语也值得扩展：那些术语往往是缩写
    （"ELBO"），补全成 "evidence lower bound" 命中率会明显提高。
    """
    return any("一" <= ch <= "鿿" for ch in query)


def expand_query(query: str, *, provider=None, use_cache: bool = True) -> list[str]:
    """把查询扩展成若干检索表述。失败时返回空列表（调用方退回原查询）。

    **不抛异常**：扩展是锦上添花，模型不可用时检索仍应正常工作。
    """
    query = (query or "").strip()
    if not query or not _needs_expansion(query):
        return []

    if use_cache:
        cached = _cache_get(query)
        if cached is not None:
            return cached

    try:
        if provider is None:
            from .llm import fast_provider

            provider = fast_provider()
        with budget.track("expand"):
            response = provider.complete(
                [{"role": "user", "content": query}],
                system=EXPAND_SYSTEM,
                # **这个额度必须给足，它决定扩展能不能出结果。**
                #
                # 实测（同一个问题连跑 6 次）：思考 6000~8700 字符，而推理型模型
                # 的思考与正文**共用**这个额度。给 2000 时 4/6 次是
                # `stop=max_tokens`、正文一个字都没有——扩展静默返回空。
                #
                # 后果被低估了很久：中文问句通向英文语料**只有扩展这一条路**
                # （全文检索匹配不上，向量通道又被判为无区分度）。一次空扩展
                # 等于这道题注定检索失败，而界面上只显示「没找到相关内容」。
                #
                # 更讽刺的是截断比成功**更贵**：4 次白跑的调用各烧 2000 token，
                # 而一次成功的调用只花约 1500。
                max_tokens=16000,
            )
        terms = _parse_terms(response.text)
    except Exception as exc:
        log.debug("查询扩展失败，将只用原查询检索：%s", exc)
        return []

    if use_cache and terms:
        _cache_put(query, terms)
    return terms


def _parse_terms(text: str) -> list[str]:
    """从模型输出里提取检索词。

    模型不总是严格按「每行一个」输出——有时用逗号分隔，有时把全部词
    写在一行里用空格隔开。与其要求它严格遵守格式（那只会失败得更频繁），
    不如宽松解析，并**在完全解析不出来时退回使用整段输出**。

    最后那个兜底很重要：即使格式不对，「一整行英文术语」本身对全文检索
    也是可用的输入。因为格式问题把好不容易拿到的术语全丢掉，是最亏的。
    """
    if not text:
        return []

    text = text.strip()
    text = re.sub(r"^\s*(?:检索词|search terms?)\s*[:：]\s*", "", text, flags=re.I)
    # 去掉常见的絮叨开头
    text = re.sub(r"^.*?(?:output|以下是|检索词)\s*[:：]?\s*", "", text, flags=re.S)
    text = text.strip()

    # 按换行、逗号、分号切成候选
    pieces = re.split(r"[\n,，;；|]+", text)
    terms: list[str] = []

    for piece in pieces:
        cleaned = piece.strip().strip("\"'“”「」*•-").strip()
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        if not cleaned or not re.search(r"[A-Za-z]", cleaned):
            continue
        # 超过 12 个词的片段多半是一句解释而不是检索词
        if len(cleaned.split()) > 12:
            continue
        terms.append(cleaned)

    if not terms:
        # 兜底：整段输出就是一行空格分隔的术语。直接拿它做检索词——
        # 全文检索本来就会分词，多词短语在这里并不需要保持完整。
        fallback = re.sub(r"[^A-Za-z0-9\s._-]+", " ", text).strip()
        if fallback and len(fallback) <= 400:
            terms = [fallback]

    # 去重，保留顺序
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        lowered = term.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(term)
    return unique[:8]


def clear_cache() -> None:
    _CACHE.clear()


def cache_size() -> int:
    return len(_CACHE)


# --------------------------------------------------------------------------
# 追问的上下文消解
# --------------------------------------------------------------------------

CONTEXTUALIZE_SYSTEM = """你在帮一个论文检索系统处理多轮对话。

用户会给你一段对话历史和一个追问。追问里往往有指代词（「它」「这篇」「上述方法」），
**单独拎出来检索不到任何东西**。你的任务是把追问改写成一句**不依赖上文也能检索**的独立问题。

要求：
- 把指代词替换成它们实际指代的对象。指代含糊时选择最可能的那个，不要提问。
- 保持原问题的意图，不要扩展成多个问题，不要回答它。
- **只输出改写后的那一句话**，不要解释、不要引号、不要编号。
- 如果追问本身已经足够独立，原样输出。

示例：

历史：用户问「DDPM 的核心贡献是什么」，助手回答了 DDPM 的贡献。
追问：「它的局限有哪些？」
输出：DDPM（Denoising Diffusion Probabilistic Models）论文的局限有哪些？"""

# 指代词。中文里这些词几乎总是指向上一轮的对象，
# 带着它们去检索等于不带任何关键词。
_CJK_ANAPHORA = re.compile(
    r"它|他们|她们|其(?!他)|该(?:论文|方法|模型|工作|文)|此(?:论文|方法|外)|"
    r"这(?:篇|个|项|种|一)|上述|前述|前面(?:提到|说)|刚才|上一(?:轮|个|篇)|该文"
)

# 追问改写的缓存。多轮对话里同一个追问常被重发（编辑后重问、刷新重试），
# 缓存能省掉重复调用。键里带上历史摘要，避免不同上下文的同一句话互相污染。
_CTX_CACHE: OrderedDict[str, str] = OrderedDict()
_CTX_CACHE_MAX = 256


def needs_context(query: str) -> bool:
    """判断这个问题是否依赖上文才能检索。

    两类：含指代词的；以及极短的追问（「那局限呢」「为什么」）——
    后者没有主语，检索无从下手。
    """
    query = (query or "").strip()
    if not query:
        return False
    if _CJK_ANAPHORA.search(query):
        return True
    # 很短、又不含任何英文实词（缩写往往就是关键词，比如「DDPM 呢」）
    return len(query) <= 12 and not re.search(r"[A-Za-z]{3,}", query)


def _history_fingerprint(history: list[dict] | None) -> str:
    """取历史里最后两轮的指纹，用于缓存键。"""
    turns = [t for t in (history or []) if t.get("role") in {"user", "assistant"}][-2:]
    parts = []
    for turn in turns:
        content = turn.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(b.get("text", "")) for b in content if isinstance(b, dict)
            )
        parts.append(f"{turn.get('role')}:{str(content)[:200]}")
    return "|".join(parts)


def contextualize_query(
    question: str,
    history: list[dict] | None,
    *,
    provider=None,
    use_cache: bool = True,
) -> str:
    """把依赖上文的追问改写成独立可检索的查询。

    **只用于检索**——生成阶段仍然拿原问题和完整历史。改写是为了让字面
    匹配够得着，不是为了改变问题的含义。

    实测过的两种做法都不行：
      * 原样检索：问「它的局限有哪些？」会取回一堆无关论文，
        「它」不携带任何可检索信息；
      * 把历史整段塞进检索：检索结果被上一轮的内容带偏，
        问新东西却一直返回旧论文。

    正确做法是先消解指代、再检索。失败时原样返回，检索照常进行。
    """
    question = (question or "").strip()
    if not question or not history:
        return question

    cache_key = f"{_history_fingerprint(history)}||{question}"
    if use_cache and cache_key in _CTX_CACHE:
        _CTX_CACHE.move_to_end(cache_key)
        return _CTX_CACHE[cache_key]

    try:
        if provider is None:
            from .llm import fast_provider

            provider = fast_provider()

        lines = []
        for turn in history[-6:]:
            content = turn.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, dict)
                )
            role = "用户" if turn.get("role") == "user" else "助手"
            lines.append(f"{role}：{str(content)[:600]}")

        with budget.track("expand"):
            response = provider.complete(
                [{"role": "user", "content": "\n".join(lines) + f"\n\n追问：{question}"}],
                system=CONTEXTUALIZE_SYSTEM,
                # 同理：指代消解也要先思考再作答，1000 会被思考吃光
                max_tokens=8000,
            )
        rewritten = _clean_rewrite(response.text)
    except Exception as exc:
        log.debug("追问改写失败，用原问题检索：%s", exc)
        return question

    if not rewritten:
        return question

    if use_cache:
        _CTX_CACHE[cache_key] = rewritten
        _CTX_CACHE.move_to_end(cache_key)
        while len(_CTX_CACHE) > _CTX_CACHE_MAX:
            _CTX_CACHE.popitem(last=False)
    return rewritten


def _clean_rewrite(text: str) -> str:
    """从模型输出里取出那一句话。

    模型偶尔会加「输出：」这类前缀或包上引号。宁可宽松解析也不要
    因为它多写了一个词就把整句丢掉。
    """
    if not text:
        return ""
    line = text.strip()
    line = re.sub(r"^\s*(?:输出|改写后|结果)\s*[:：]\s*", "", line)
    line = line.strip().strip("\"'“”「」").strip()
    # 只取第一行——万一它多写了说明，第一行才是改写结果
    line = line.splitlines()[0].strip() if line else ""
    return line[:400]


__all__ = [
    "cache_size",
    "clear_cache",
    "contextualize_query",
    "expand_query",
    "needs_context",
]
