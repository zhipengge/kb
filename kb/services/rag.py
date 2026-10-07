"""检索增强问答。

**这个模块的核心承诺是「引用可验证」**，其余设计都围绕它：

1. 检索到的每条片段在提示里带一个编号（``[1]``、``[2]``…），映射关系由
   服务端持有。**不让模型自己写论文标题或 id**——它写不对，而且写错了
   你也看不出来。
2. 模型回答时只能引用这些编号。
3. 拿到回答后**逐条校验**：编号对应的片段里，是否真的出现了回答中引用的
   那句话。验不过的标记为「未验证」，界面上会显示出来。

第 3 步是关键。没有它，「引用」只是装饰——模型可以随便标个 [3]，
而用户不会去核对。有了它，编造的引用会被明确标出来。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace

from . import budget
from .search import SearchHit, search
from .websearch import WebResult

log = logging.getLogger(__name__)

# 引用标记：模型可能写 [1]、[1,2]、[1][2]
_CITATION_MARK = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

RAG_SYSTEM = """你在帮一位研究者查阅他自己的论文知识库。

回答要求：

- **只依据给定的资料回答**。资料里没有的内容，直接说「知识库中没有找到相关信息」，
  不要用你的先验知识补上——这个知识库的价值就在于可信。
- 用中文回答，技术术语保留英文原文。
- 如果不同资料的结论互相矛盾，**指出来**，不要挑一个当成正确答案。
- 简洁。研究者要的是信息，不是客套。

**引用格式（必须严格遵守）：**

每个论断后面标编号，并在编号后附上**原文中的一小段原话**，用引号包起来：:

    Transformer 完全依赖注意力机制，不用循环和卷积 [1]"relying entirely on an
    attention mechanism to draw global dependencies"。

引文的要求：

- 必须是资料里**逐字存在**的片段，保持资料原本的语言（通常是英文），
  不要翻译、不要改写、不要补全。
- 长度 5-20 个词即可，够定位就行。
- 引文是给人核对用的。宁可引短一点，也不要凭印象写一句「大概是这个意思」的话——
  那样会让核对失效。

资料里没有合适原话可引时，就**不要标编号**，直接说明该处没有依据。"""


# 联网搜索工具的说明。措辞要**保守**——它决定了模型什么时候会跑出去上网。
# 写成「需要更多信息时可以搜索」会让它频繁外呼，而知识库能答的问题
# 上网查既慢又不可靠（网上可能有更新但更差的版本）。
WEB_SEARCH_TOOL_SYSTEM = """

**联网搜索的使用条件（严格）：**

你有一个 web_search 工具，可以查互联网。**默认不要用它。**

只有同时满足以下两条时才用：
1. 知识库里**确实没有**相关信息（你已经检索过了，资料里没有）；
2. 问题本身指向知识库之外的东西——某篇论文被谁引用了、某个方向最新进展、
   某个框架的官方文档怎么写、某个报错别人怎么解决的。

**不要**为了「补充细节」「验证一下」而搜索。知识库里有依据的问题，
就基于知识库回答并说明依据在哪。

搜索结果**不是知识库资料**，可靠度不同。用到时必须：
- 明确标注「以下来自联网检索，未经知识库核对」；
- 引用时用 `[W1]`、`[W2]` 这样的编号（区别于知识库的 `[1]`、`[2]`）。
"""


WEB_TOOL_NAME = "web_search"

# 联网轮数上限。给 2 轮是因为「先查领域现状、再查具体那篇」是常见且合理的两步；
# 再多就基本是在兜圈子了，而每一轮都要用户多等十几秒。
MAX_TOOL_ROUNDS = 2

WEB_TOOL_SCHEMA = {
    "name": WEB_TOOL_NAME,
    "description": (
        "联网检索。**仅在知识库确实没有相关信息时使用**，"
        "不要用它来补充或验证知识库里已有的内容。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "检索词。用**英文**效果最好（学术库与 GitHub 都以英文为主）。"
                    "不要在检索词里写「最新」「2024 年」这类词，用具体术语。\n"
                    "**只给平实的关键词，不要用搜索引擎语法。**"
                    "这里接的是结构化 API（OpenAlex / Crossref / arXiv / GitHub / "
                    "HackerNews），不是搜索引擎：``site:``、引号、``OR``、``-`` "
                    "这类操作符它们不认，会把结果清空。"
                    "想要某个来源的内容，改 ``kind`` 而不是写在检索词里——"
                    "实测同一个问题，平实关键词能出 4 类来源，"
                    "加上 ``site:news.ycombinator.com`` 就只剩 1 类。"
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["paper", "code", "web", "auto"],
                "description": (
                    "查哪一类来源。**选错只会浪费一轮**，但知道各自能查什么能省掉这轮：\n"
                    "- paper：学术文献（OpenAlex / Crossref / arXiv / Semantic Scholar）。"
                    "问「这个方向有哪些工作」「某篇被谁引用了」「某个方法的研究现状」时用。\n"
                    "- code：GitHub **仓库**搜索，匹配的是**仓库名和描述**，"
                    "不是仓库里的代码。所以它适合问「某方法有没有开源实现」"
                    "「某方向的知名仓库」，**不适合**问「某函数怎么用」——"
                    "那类问题请选 web。\n"
                    "- web：技术社区与讨论（HackerNews）+ 通用网页（需配置 Tavily Key）。"
                    "问「外界怎么评价这项工作」「有没有人复现失败」「API 用法、"
                    "报错怎么解决」时用。\n"
                    "- auto：不确定时用，会把启用的几类都查一遍。"
                ),
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def _web_tool_spec(settings) -> tuple[list[dict], tuple[str, ...]]:
    """决定这次是否给模型联网工具，以及启用哪些类别。

    返回 (工具列表, 类别)。工具列表为空表示不提供——这时系统提示里也**不会**
    提联网，否则模型会以为自己能搜却搜不了。
    """
    if not settings.get("websearch.enabled"):
        return [], ()

    kinds: list[str] = []
    if settings.get("websearch.academic"):
        kinds.append("paper")
    if settings.get("websearch.code"):
        kinds.append("code")
    # web 这一类现在不只包含 Tavily：HackerNews 与 Stack Exchange 也在里面，
    # 而它们不需要 Key。所以判定条件不能再只看 Tavily——
    # 否则「没配 Tavily」会连累这两个免 Key 的源一起用不了，
    # 而那正是「通用网页搜索没有 Key」时唯一还能用的东西。
    #
    # 取密钥必须用 has_secret()：secret 存的是密文，get() 拿到的是密文本身。
    if settings.get("websearch.discussions") or settings.has_secret(
        "websearch.tavily_api_key"
    ):
        kinds.append("web")

    if not kinds:
        return [], ()
    return [WEB_TOOL_SCHEMA], tuple(kinds)


def _run_web_search(query: str, settings, available: tuple[str, ...], kind: str = "auto"):
    """执行联网检索。``kind`` 由模型指定，越界时回退到全部可用类别。"""
    from .websearch import search as web_search

    requested = (kind or "auto").strip().lower()
    if requested == "auto" or requested not in {"paper", "code", "web"}:
        kinds = available
    elif requested in available:
        kinds = (requested,)
    else:
        # 模型要了一类没启用的来源（比如没配 Tavily 却要 web）。
        # 返回明确的说明而不是空列表，让它能换个类别重试。
        reason = {
            "web": "通用网页搜索未启用（需要在设置里配置 Tavily API Key）",
        }.get(requested, f"{requested} 类检索未启用")
        return [], [reason]

    return web_search(query, settings=settings, kinds=kinds)


# 引用的校验状态。三态而不是布尔值——「没给引文」和「引文对不上」
# 的可疑程度差得很远：前者只是没用上校验能力，后者可能意味着这段内容
# 是编的。用同一个 False 表示两者，会让用户对校验结果失去分辨力。
CHECK_VERIFIED = "verified"      # 引文确实出现在被引片段里
CHECK_UNVERIFIED = "unverified"  # 模型没给逐字引文，无从校验
CHECK_MISMATCHED = "mismatched"  # 给了引文，但在被引片段里找不到 —— 最可疑


@dataclass
class Citation:
    """一条引用。"""

    marker: int
    chunk_id: str
    paper_id: str | None = None
    note_id: str | None = None
    title: str = ""
    section_path: str | None = None
    page_from: int | None = None
    page_to: int | None = None
    quote: str = ""
    check: str = CHECK_UNVERIFIED

    @property
    def verified(self) -> bool:
        return self.check == CHECK_VERIFIED

    def to_dict(self) -> dict:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "note_id": self.note_id,
            "title": self.title,
            "section_path": self.section_path,
            "page_from": self.page_from,
            "page_to": self.page_to,
            "locator": self.locator,
            "quote": self.quote,
            "check": self.check,
            "verified": self.verified,
        }

    @property
    def locator(self) -> str:
        bits = []
        if self.section_path:
            bits.append(f"§{self.section_path}")
        if self.page_from:
            if self.page_to and self.page_to != self.page_from:
                bits.append(f"p.{self.page_from}-{self.page_to}")
            else:
                bits.append(f"p.{self.page_from}")
        return " ".join(bits)


@dataclass
class RagAnswer:
    """一次知识库问答的结果。"""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[SearchHit] = field(default_factory=list)
    model: str = ""
    tokens_used: int = 0
    elapsed_ms: int = 0
    grounded: bool = True  # 是否所有引用都通过了校验（无「对不上」的）
    error: str | None = None
    # 联网检索到的来源。**与 citations 分开**：知识库引用能逐字核对原文，
    # 联网结果不能，两者的可靠度不是一个量级，混在一个数组里
    # 会让人以为它们同等可信。
    web_citations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "web_citations": self.web_citations,
            "grounded": self.grounded,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


# --------------------------------------------------------------------------
# 上下文组装
# --------------------------------------------------------------------------


def build_context(hits: list[SearchHit]) -> str:
    """把检索结果编号后拼成给模型的资料。

    编号与顺序都保持稳定：提示缓存是前缀匹配，同一批资料每次拼出来
    必须逐字一致，否则缓存永远命不中。
    """
    blocks: list[str] = []
    for index, hit in enumerate(hits, 1):
        source = hit.paper_title or hit.note_title or "未命名"
        locator = hit.locator or ""
        header = f"[{index}] 来源：{source}"
        if locator:
            header += f"（{locator}）"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


def _normalize_for_match(text: str) -> str:
    """归一化文本用于引文比对。

    去空白、转小写、统一标点。论文里换行与空格的位置很随意，
    直接字符串比对会把大量正确的引用判成「未验证」。
    """
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。、；：""''（）《》\\[\\](){}.,;:!?\"']", "", text)
    return text


# 引用后面的逐字引文。模型被要求写成 [1]"verbatim quote" 的形式，
# 引号可能是英文双引号、中文引号或单引号。
_FOLLOWING_QUOTE = re.compile(
    r"""\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]\s*[「“"']([^」”"']{8,300})[」”"']"""
)


def verify_citations(answer: str, hits: list[SearchHit]) -> list[Citation]:
    """解析回答里的引用，并用「模型给出的逐字引文」来校验。

    **为什么要求模型附引文，而不是拿它自己的话来比对。**

    模型通常用中文回答，而资料原文是英文。拿中文句子去英文原文里找匹配，
    永远匹配不上——每条正确的引用都会被判成假的，校验就失去了意义。

    要求模型附上原文的逐字片段之后，比对变成「同语言的字符串包含」，
    既准确又便宜。这也是 Anthropic Citations API 的做法：
    让模型给出可核对的原文锚点，而不是让它自己声称「我参考了 [3]」。

    三种状态：
      * ``verified=True``   —— 引文确实出现在被引片段里
      * ``verified=False`` 且 ``quote`` 为空 —— 模型没给引文，无从校验
      * ``verified=False`` 且 ``quote`` 非空 —— 给了引文但对不上，**这最可疑**
    """
    citations: list[Citation] = []
    seen: set[int] = set()

    # 先收集「标记 -> 紧跟其后的逐字引文」
    quote_by_marker: dict[int, str] = {}
    for match in _FOLLOWING_QUOTE.finditer(answer):
        for raw in match.group(1).split(","):
            try:
                marker = int(raw.strip())
            except ValueError:
                continue
            quote_by_marker.setdefault(marker, match.group(2).strip())

    for match in _CITATION_MARK.finditer(answer):
        for raw in match.group(1).split(","):
            try:
                marker = int(raw.strip())
            except ValueError:
                continue
            if marker in seen or not (1 <= marker <= len(hits)):
                continue
            seen.add(marker)

            hit = hits[marker - 1]
            quote = quote_by_marker.get(marker, "")

            if quote:
                # 模型给了原文片段：能确凿地判定「对得上」还是「对不上」
                check = (
                    CHECK_VERIFIED if _quote_supported(quote, hit.text)
                    else CHECK_MISMATCHED
                )
            else:
                # 没给引文。退回「用回答里那句话去比对」——只在同语言时才有意义，
                # 跨语言必然失败。判不出来就标为「无从校验」，
                # **不能因为判不出来就说它错**，那会让正确的引用背黑锅。
                prefix = answer[: match.start()].rstrip()
                sentence = re.split(r"[。！？\n]", prefix)[-1][-300:]
                sentence = _CITATION_MARK.sub("", sentence).strip()
                check = (
                    CHECK_VERIFIED if sentence and _quote_supported(sentence, hit.text)
                    else CHECK_UNVERIFIED
                )

            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=hit.chunk_id,
                    paper_id=hit.paper_id,
                    note_id=hit.note_id,
                    title=hit.paper_title or hit.note_title or "",
                    section_path=hit.section_path,
                    page_from=hit.page_from,
                    page_to=hit.page_to,
                    quote=quote,
                    check=check,
                )
            )

    citations.sort(key=lambda c: c.marker)
    return citations


def _quote_supported(quote: str, source: str, *, threshold: float = 0.42) -> bool:
    """判断引文是否被原文支持。

    用「引文的关键词有多少出现在原文里」而不是整句比对：模型转述时
    会调整措辞，整句比对过严；但只查一个词又过松（任何句子都能碰上一个词）。
    阈值取 0.42 是实测下来「转述能过、编造过不了」的位置。
    """
    if not quote or not source:
        return False

    normalized_source = _normalize_for_match(source)
    if not normalized_source:
        return False

    normalized_quote = _normalize_for_match(quote)
    if not normalized_quote:
        return False

    # 整句直接命中，最理想
    if normalized_quote in normalized_source:
        return True

    # 退一步：按 4 字滑窗算命中比例
    window = 4
    if len(normalized_quote) < window:
        return normalized_quote in normalized_source

    grams = [
        normalized_quote[i : i + window]
        for i in range(0, len(normalized_quote) - window + 1, 2)
    ]
    if not grams:
        return False
    hits = sum(1 for gram in grams if gram in normalized_source)
    return (hits / len(grams)) >= threshold


# --------------------------------------------------------------------------
# 问答
# --------------------------------------------------------------------------


def _retrieval_query(question: str, history: list[dict] | None, provider=None) -> str:
    """检索用的查询。

    多轮对话里的追问（「它的局限有哪些？」）单独拿去检索什么都取不到——
    「它」不携带任何可检索信息。实测这种情况会返回一堆完全无关的论文，
    而生成侧因为拿到了历史，答案本身却是对的：**检索与生成用了两个不同的
    问题**，这是最容易被忽略的错配。

    另一条弯路也验证过：把整段历史塞进检索，结果是检索被上一轮内容带偏，
    问新东西一直返回旧论文。正确做法是先把指代消解成独立问题，再检索；
    生成仍然用原问题和完整历史。
    """
    question = (question or "").strip()
    if not question or not history:
        return question
    from . import query_expand as qe

    if not qe.needs_context(question):
        return question
    return qe.contextualize_query(question, history, provider=provider)


def _build_messages(
    question: str,
    context: str,
    history: list[dict] | None,
    question_blocks: list[dict] | None = None,
) -> list[dict]:
    """组装送给模型的消息序列。

    **一次性问答与流式问答共用这一份**，不是各自拼一遍。实测教训：
    流式那条路径自己拼消息，于是完全忽略 ``history``——界面上的多轮追问
    永远失忆，而接口返回 200、没有任何报错。两条路径共用同一个组装函数，
    这类分叉就不可能再发生。
    """
    messages: list[dict] = []
    turns = [t for t in (history or [])[-6:] if t.get("role") in {"user", "assistant"}]

    for index, turn in enumerate(turns):
        content = turn.get("content")
        if not content:
            continue
        role = turn["role"]

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content)[:4000]})
            continue

        # 内容块数组（图片、文档等）要原样透传，不能 `str()`。
        #
        # 以前这里无条件 `str(content)[:4000]`：图片块会被字符串化成
        # "{'type': 'image', 'source': {...一万多字符的 base64...}}" 的一坨文本。
        # 模型看不懂，还白占了历史预算。多模态会话里「上一轮我给你看的那张图」
        # 就是在这里静默丢掉的。
        #
        # 但也不能把每一轮的图都带上：一张图动辄上万 token，六轮下来足以
        # 把上下文撑爆。所以**只保留最近一轮的图片**，更早的换成占位文字——
        # 指代（「这张图里…」）几乎总是指最近那张。
        is_latest = index == len(turns) - 1
        blocks: list[dict] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image" and not is_latest:
                blocks.append({"type": "text", "text": "［此前提供的一张图片，已省略］"})
            else:
                blocks.append(block)
        messages.append({"role": role, "content": blocks})

    # 提问本身也可能是内容块（带图/带附件）。检索只用文字部分，
    # 但生成时要看到完整的块，否则「这张图里有什么」会退化成「资料里有什么」。
    if question_blocks:
        messages.append(
            {
                "role": "user",
                "content": [*question_blocks,
                            {"type": "text",
                             "text": f"资料：\n\n{context}\n\n---\n\n请基于以上资料回答上面的问题。"}],
            }
        )
    else:
        messages.append(
            {"role": "user", "content": f"资料：\n\n{context}\n\n---\n\n问题：{question}"}
        )
    return messages


def answer(
    question: str,
    *,
    limit: int | None = None,
    filters: dict | None = None,
    history: list[dict] | None = None,
    question_blocks: list[dict] | None = None,
    extra_system: str | None = None,
    provider=None,
    ctx=None,
    ref: str | None = None,
) -> RagAnswer:
    """基于知识库回答问题。

    ``history`` 是多轮对话的上下文（不含本次问题），用于让「它」这种
    指代能对上。但**检索只用当前问题**——把历史也塞进检索会让
    「那它的局限呢」这种追问检索到一堆无关内容。
    """
    import time

    started = time.perf_counter()

    from flask import current_app

    settings = current_app.extensions["kb_settings"]

    question = (question or "").strip()
    if not question:
        return RagAnswer(answer="", error="问题为空")

    # ---- 检索 ----
    top_k = limit or int(settings.get("retrieval.top_k"))
    hits = search(
        _retrieval_query(question, history, provider),
        limit=top_k,
        filters=filters or {},
        group_by_paper=bool(settings.get("retrieval.group_by_paper")),
        rrf_k=int(settings.get("retrieval.rrf_k")),
    )

    if not hits:
        return RagAnswer(
            answer=(
                "知识库里没有找到与这个问题相关的内容。\n\n"
                "可能的原因：论文还没完成索引、问的是知识库未覆盖的领域，"
                "或者换个更具体的说法能检索到。"
            ),
            model="",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    if ctx is not None:
        ctx.progress(0.4, f"检索到 {len(hits)} 条相关内容")

    # ---- 生成 ----
    context = build_context(hits)
    system = RAG_SYSTEM if not extra_system else f"{RAG_SYSTEM}\n\n{extra_system}"

    messages = _build_messages(question, context, history, question_blocks)

    if provider is None:
        from .llm import chat_provider

        provider = chat_provider()

    # 联网工具。对外接口（/api/v1/ask）走的就是这条路径，
    # 所以这里同样要有——否则网页端能联网、agent 调接口却不行，
    # 同一套知识库给出两种能力，是最难解释的那种不一致。
    web_tools, web_kinds = _web_tool_spec(settings)
    if web_tools:
        system = system + WEB_SEARCH_TOOL_SYSTEM
    web_results: list[WebResult] = []

    try:
        # 每一轮工具调用都是独立的计费调用，所以三次 provider.complete
        # 各自带上 track——记出来的条数才和账单上的条数对得上。
        with budget.track("ask", ref=ref):
            response = provider.complete(messages, system=system, tools=web_tools or None)
        for _round in range(MAX_TOOL_ROUNDS):
            calls = [c for c in response.tool_calls if c.get("name") == WEB_TOOL_NAME]
            if not calls:
                break
            messages.append({"role": "assistant", "content": response.raw_content})
            tool_results = []
            for call in calls:
                payload = call.get("input") or {}
                query = str(payload.get("query") or question)[:300]
                kind = str(payload.get("kind") or "auto")
                found, errors = _run_web_search(query, settings, web_kinds, kind)
                offset = len(web_results)
                web_results.extend(found)
                blocks = [
                    item.as_context(f"W{offset + index}")
                    for index, item in enumerate(found, 1)
                ]
                if not blocks:
                    detail = "；".join(errors) if errors else "没有返回结果"
                    blocks = [f"（联网检索没有结果：{detail}）"]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.get("id"),
                        "content": "\n\n".join(blocks),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            with budget.track("ask", ref=ref):
                response = provider.complete(messages, system=system, tools=web_tools)
        else:
            # 步数用光——和流式那条一样必须收尾，否则调用方拿到的是空回答。
            # 对外接口上这个问题更隐蔽：agent 拿到 `answer: ""` 只会以为
            # 「知识库没查到这个」，而不会想到是工具把步数烧光了。
            with budget.track("ask", ref=ref):
                response = provider.complete(messages, system=system, tools=None)
            if not response.text.strip():
                response = replace(  # type: ignore[assignment]
                    response,
                    text="（模型没有给出回答。可以把问题问得更具体，或减少需要查的内容。）",
                )
    except Exception as exc:
        log.exception("生成回答失败")
        return RagAnswer(
            answer="",
            contexts=hits,
            model=getattr(provider, "model", ""),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            error=f"模型调用失败：{exc}",
        )

    if ctx is not None:
        ctx.progress(0.85, "校验引用")

    # ---- 校验引用 ----
    # 清理被当成正文吐出来的工具调用标记，再拿去校验引用——
    # 标记里可能夹着引用编号，先清掉才不会把噪声算进校验
    response_text = strip_tool_markup(response.text)
    citations = verify_citations(response_text, hits)
    # 「无从校验」不算不通过——只对「引文对不上」报警
    grounded = all(c.check != CHECK_MISMATCHED for c in citations) if citations else True

    if not grounded:
        bad = sum(1 for c in citations if not c.verified)
        log.warning("回答中有 %d 条引用未能通过校验", bad)

    return RagAnswer(
        answer=response_text,
        citations=citations,
        contexts=hits,
        model=response.model,
        tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        grounded=grounded,
        web_citations=[
            {"marker": f"W{index}", **item.to_dict()}
            for index, item in enumerate(web_results, 1)
        ],
    )


def answer_streaming(question: str, **kwargs):
    """流式问答。

    产出 ``{"type": "sources"|"text"|"done"|"error", ...}`` 事件。
    先发 sources（检索结果），让前端立刻能显示「在查这些论文」，
    不必等模型开始输出——模型思考可能要十几秒。
    """
    from flask import current_app

    settings = current_app.extensions["kb_settings"]

    question = (question or "").strip()
    if not question:
        yield {"type": "error", "message": "问题为空"}
        return

    filters = kwargs.get("filters") or {}
    top_k = kwargs.get("limit") or int(settings.get("retrieval.top_k"))
    # 账本上把这几次调用归到哪个会话名下。没有会话时留空——
    # 宁可不归因，也不要编一个 ID 出来。
    ref = kwargs.get("ref")

    hits = search(
        _retrieval_query(question, kwargs.get("history"), kwargs.get("provider")),
        limit=top_k,
        filters=filters,
    )
    if not hits:
        yield {
            "type": "text",
            "text": "知识库里没有找到与这个问题相关的内容。",
        }
        yield {"type": "done", "citations": [], "grounded": True}
        return

    yield {
        "type": "sources",
        "sources": [
            {
                "marker": index,
                "chunk_id": hit.chunk_id,
                "paper_id": hit.paper_id,
                "title": hit.paper_title or hit.note_title,
                "locator": hit.locator,
            }
            for index, hit in enumerate(hits, 1)
        ],
    }

    context = build_context(hits)
    provider = kwargs.get("provider")
    if provider is None:
        from .llm import chat_provider

        provider = chat_provider()

    # 与一次性问答共用同一份组装逻辑。带上 history 与 question_blocks，
    # 多轮追问和带图提问在流式路径上才成立。
    messages = _build_messages(
        question,
        context,
        kwargs.get("history"),
        kwargs.get("question_blocks"),
    )
    accumulated: list[str] = []
    web_results: list[WebResult] = []

    # 联网工具是否可用。一个后端都用不了时索性不把工具给模型——
    # 给了它却调不通，只会得到「我搜了一下但没找到」这种更让人困惑的答案。
    web_tools, web_kinds = _web_tool_spec(settings)
    system = RAG_SYSTEM + (WEB_SEARCH_TOOL_SYSTEM if web_tools else "")

    # --- 生成（含工具调用循环）---
    #
    # 循环而不是一次调用：模型可能先要求联网、看到结果后还要再查一次、
    # 最后才作答。轮数设上限是防止它陷入「搜了又问、问了又搜」——
    # 那会一直烧预算，而用户只是想要一个答案。
    for _round in range(MAX_TOOL_ROUNDS + 1):
        final = None
        with budget.track("ask", ref=ref):
            for event in provider.stream_text(messages, system=system, tools=web_tools or None):
                kind = event["type"]
                if kind == "text":
                    accumulated.append(event["text"])
                    yield event
                elif kind == "thinking":
                    yield event
                elif kind == "error":
                    yield event
                    return
                elif kind == "done":
                    final = event["response"]

        calls = [c for c in (final.tool_calls if final else []) if c.get("name") == WEB_TOOL_NAME]
        if not calls:
            break

        messages.append({"role": "assistant", "content": final.raw_content})
        tool_results = []
        for call in calls:
            payload = call.get("input") or {}
            query = str(payload.get("query") or question)[:300]
            kind = str(payload.get("kind") or "auto")
            yield {"type": "web_searching", "query": query, "kind": kind}

            found, errors = _run_web_search(query, settings, web_kinds, kind)
            # 编号用 W1、W2…与知识库的 [1]、[2] 分开：
            # 两类来源的可靠度不同，混用编号会让人以为出处是一回事
            offset = len(web_results)
            web_results.extend(found)

            blocks = [
                item.as_context(f"W{offset + index}")
                for index, item in enumerate(found, 1)
            ]
            if not blocks:
                detail = "；".join(errors) if errors else "没有返回结果"
                blocks = [f"（联网检索没有结果：{detail}）"]
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.get("id"),
                    "content": "\n\n".join(blocks),
                }
            )
            yield {
                "type": "web_sources",
                "results": [item.to_dict() for item in found],
                "errors": errors,
            }

        messages.append({"role": "user", "content": tool_results})
    else:
        # **循环跑满还没出答案，必须收尾。**
        #
        # for...else：只有「每一轮都在调工具」才会走到这里——也就是步数用光。
        # 不收尾的话，用户看到的就是「我来查一下」加上几轮工具调用，
        # **他问的问题一个字都没答**，界面上也没有任何解释。
        #
        # 收尾的办法是**摘掉工具再问一次**，逼模型拿手里的结果作答。
        # 工具必须传 None 而不是 []——网关对空数组会 400。
        yield {
            "type": "notice",
            "message": "工具调用已达上限，下面是基于已查到内容的回答",
        }
        try:
            # 这一轮**故意不给工具**（逼模型作答），而网关恰恰在这里最容易
            # 把工具调用标记当正文吐出来——模型还想查，却没有工具可调。
            # 所以这段必须过过滤器。
            markup = ToolMarkupFilter()
            with budget.track("ask", ref=ref):
                for event in provider.stream_text(messages, system=system, tools=None):
                    if event["type"] == "text":
                        visible = markup.feed(event["text"])
                        if not visible:
                            continue
                        accumulated.append(visible)
                        yield {**event, "text": visible}
                    elif event["type"] == "thinking":
                        yield event
                    elif event["type"] == "error":
                        yield event
                        break
                # 把过滤器扣住的尾巴放出来，否则回答末尾会少几个字符
                tail = markup.flush()
                if tail:
                    accumulated.append(tail)
                    yield {"type": "text", "text": tail}
        except Exception as exc:
            log.warning("收尾调用失败：%s", exc)

        if not "".join(accumulated).strip():
            # 逼了一次还是不说话，就明说，别让用户对着空白猜
            fallback = "（模型没有给出回答。可以试着把问题问得更具体，或减少它需要查的东西。）"
            accumulated.append(fallback)
            yield {"type": "text", "text": fallback}

    full = strip_tool_markup("".join(accumulated))
    citations = verify_citations(full, hits)
    yield {
        "type": "done",
        "citations": [c.to_dict() for c in citations],
        "web_citations": [
            {"marker": f"W{index}", **item.to_dict()}
            for index, item in enumerate(web_results, 1)
        ],
        "grounded": all(c.check != CHECK_MISMATCHED for c in citations) if citations else True,
        "tokens_used": 0,
    }


# 全角竖线。某些网关（实测 DeepSeek 的 Anthropic 兼容端点）在模型"
# 想调用工具、却没有可用的工具声明时，会把**工具调用标记当正文吐出来**，
# 形如：全角竖线 x2 + DSML + 全角竖线 x2 + invoke name="web_search" ...
#
# 出现这种输出的典型场景：工具循环用光步数后，收尾那一轮**故意不带工具**
# （见 answer_streaming 的 for...else），模型却还想再查一次，于是把调用
# 意图编码成文本。不清理的话，用户会看到一整段尖括号标记。
def _first_marker(text: str) -> int:
    """第一个工具调用标记的位置，没有则返回 -1。"""
    index = text.find(_DSML)
    lowered = text.lower()
    for opener in ("<invoke", "<parameter", "<tool_call", "<function_call"):
        found = lowered.find(opener)
        if found != -1 and (index == -1 or found < index):
            index = found
    return index


def strip_tool_markup(text: str) -> str:
    """去掉被当成正文吐出来的工具调用标记。

    **从第一个标记起，把后面的内容整段丢掉**，而不是只删标记本身、留下正文。

    理由是实测出来的：这段标记的内部，定界符会出现**很多次**——它是每一层
    标签之间的分隔符，不是一对开闭括号。按「见到定界符就切换丢弃/保留」写，
    会在层层标签之间来回翻转，把标签中间的内容当正文漏出去
    （第一版就是那么错的，实测漏出了 ``invoke name=...`` 这类片段）。

    而它出现时的真实形态是：模型整条回答都在尝试调用工具，里面并没有夹带
    任何有用的话。所以整段丢掉既简单又准确；丢完若什么都不剩，
    调用方会走「模型没有给出回答」的兜底，不会让用户对着空白猜。
    """
    if not text:
        return ""
    index = _first_marker(text)
    if index == -1:
        return text
    return text[:index].rstrip()


# 网关把工具调用当正文吐出来时用的分隔符（全角竖线夹着 DSML）。
# 它同时是**开始和结束**的定界符，因此可以拿来当状态机的开关。
_DSML = "｜｜DSML｜｜"

class ToolMarkupFilter:
    """流式路径的标记过滤器。

    **为什么不能只在最后清理。** 流式回答是一段段 yield 给前端的，前端边收边渲染；
    等最后再 replace 已经晚了——用户屏幕上那段标记早就出现了。

    麻烦在于标记会被切碎：``｜｜DS`` 和 ``ML｜｜`` 可能落在相邻两个 chunk 里。
    所以这里是个**状态机**：一旦看到起始定界符就进入「丢弃」状态，
    一直丢到遇上下一个定界符加 ``>`` 为止，中途来的数据一律不发。

    第一版不是这么写的——当时只判断「缓冲区结尾是不是定界符的前缀」，
    结果标记一旦超过前缀长度就不再匹配，半截标记照样发了出去。
    实测那条路在「跨 chunk」这一项上是失败的，所以改成状态机。
    """

    def __init__(self) -> None:
        self._buf = ""
        self._dropping = False

    def _safe_prefix(self) -> int:
        """不进入标记状态时，结尾有多少字符要扣住以防定界符被切开。"""
        # 只可能被切开的是定界符的前缀，以及它前面的那个 '<'
        for size in range(min(len(self._buf), len(_DSML) + 1), 0, -1):
            tail = self._buf[-size:]
            if _DSML.startswith(tail) or ("<" + _DSML).startswith(tail):
                return size
        return 0

    def feed(self, delta: str) -> str:
        """吃进一个 delta，返回**可以安全发给前端**的部分。"""
        if not delta:
            return ""
        if self._dropping:
            # 已经进入标记，这一轮剩下的全部丢弃
            return ""

        self._buf += delta
        index = _first_marker(self._buf)
        if index != -1:
            # 标记开始：它之前的内容照发，之后的一律不要了
            head = self._buf[:index]
            if head.endswith("<"):
                head = head[:-1]
            self._buf = ""
            self._dropping = True
            return head

        # 没看到标记，但结尾可能是被切开的标记前缀——扣住不发，等下个 chunk
        hold = self._safe_prefix()
        cut = len(self._buf) - hold
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return ready

    def flush(self) -> str:
        """收尾。

        仍在丢弃状态说明这段标记没写完就结束了。**仍然不发**——把它当正文
        放出去才是真正的错误（用户会看到半截尖括号）。丢掉顶多少几个字符，
        而且调用方还有「模型没有给出回答」的兜底。
        """
        rest, self._buf = self._buf, ""
        if self._dropping:
            self._dropping = False
            return ""
        self._dropping = False
        return rest


__all__ = [
    "CHECK_MISMATCHED",
    "CHECK_UNVERIFIED",
    "CHECK_VERIFIED",
    "Citation",
    "RagAnswer",
    "answer",
    "answer_streaming",
    "build_context",
    "strip_tool_markup",
    "verify_citations",
]
