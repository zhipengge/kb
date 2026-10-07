"""给外部 agent 用的工具表。

**这里是唯一的一份工具声明。** MCP 的 ``tools/list``、agent 指南里的工具清单、
以后的其它接入面，全部从这张表生成。手抄第二份的后果不是「多维护一处」，
而是两份会**各自漂移**：改了实现忘了改文档，调用方按文档调会失败；
删了工具文档还留着，模型会去调一个不存在的工具，然后编一个结果出来。
所以这张表既描述工具，也是工具本身——没有第二处需要同步。

**全部只读。** 这不是偷懒，是刻意的边界：MCP 的调用方通常是模型，
而模型没有「后悔」的能力。给它一个 `delete_note`，它会在某次误解指令时
真的删掉。要写就走 REST 接口——那边有 ``write`` 作用域的 Key、乐观锁、
字段白名单，而且**由人决定要不要给**。要读的东西这里全都有。

每个工具的 ``description`` 是写给**模型**看的，所以它必须回答两个问题：
「什么时候该用我」和「我返回什么」。只写「检索论文」是不够的——
模型在 ``search`` 和 ``ask`` 之间选择时，需要知道前者是检索（快、只顺带一次轻量调用）、
后者是端到端生成（慢、贵得多）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class ToolError(Exception):
    """工具执行失败。消息面向调用方。"""


@dataclass(frozen=True)
class AgentTool:
    """一个可供 agent 调用的工具。"""

    name: str
    title: str
    description: str
    properties: dict[str, Any]
    handler: Callable[[dict], Any]
    required: list[str] = field(default_factory=list)
    # 成本说明。非空时会追加到 description 末尾。
    #
    # 原本这里是个 `paid: bool`，但布尔值会撒谎：`search` 本身不调模型，
    # 却默认会触发一次**查询扩展**的大模型调用——实测单次约 1.6 秒、几百 token。
    # 标成「免费」会让调用方（模型）放心地连发十几次检索，账单和对延迟的
    # 预期都会落空。所以改成一段**说清楚什么情况下花多少**的文字。
    cost: str = ""

    @property
    def input_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": self.properties,
            "additionalProperties": False,
        }
        if self.required:
            schema["required"] = self.required
        return schema

    def to_spec(self) -> dict[str, Any]:
        """MCP ``tools/list`` 里的一项。"""
        spec: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
            # 全是只读工具，把这一点作为注解显式声明。
            # 客户端据此可以省掉「要不要确认」的询问。
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        }
        if self.cost:
            spec["description"] = f"{spec['description']}\n\n【成本】{self.cost}"
        return spec


# --------------------------------------------------------------------------
# 工具实现
# --------------------------------------------------------------------------


def _search(args: dict) -> dict:
    from .search import search

    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("query 不能为空")

    filters: dict[str, Any] = {}
    if args.get("paper_ids"):
        filters["paper_ids"] = list(args["paper_ids"])
    if args.get("tag_ids"):
        filters["tag_ids"] = list(args["tag_ids"])

    hits = search(
        query,
        limit=int(args.get("limit") or 8),
        filters=filters,
        group_by_paper=bool(args.get("group_by_paper", False)),
    )
    return {
        "query": query,
        "count": len(hits),
        "results": [
            {
                "chunk_id": hit.chunk_id,
                "paper_id": hit.paper_id,
                "note_id": hit.note_id,
                "title": hit.paper_title or hit.note_title,
                "kind": hit.kind,
                # locator 已经拼好（形如「§3 Method p.4」），可以直接当出处写进回答
                "locator": hit.locator,
                # section_path 与 page_from 单独给出来，是为了让「接着读这一节」
                # 成为一个机械动作：把它们原样传给 read_paper 就行。
                # 只给 locator 的话，调用方得去解析「§X p.N」这个人类可读的串——
                # 让模型做字符串切分是自找麻烦，切错了还不报错，只是读错地方。
                "section_path": hit.section_path,
                "page_from": hit.page_from,
                "text": hit.text,
            }
            for hit in hits
        ],
        "hint": (
            "locator 可直接用作引用出处。要读命中片段附近的更多内容，"
            "把这一条的 paper_id 与 section_path（或 page_from）原样传给 read_paper。"
        ),
    }


def _read_paper(args: dict) -> dict:
    from .papers import paper_text

    paper_id = str(args.get("paper_id") or "").strip()
    if not paper_id:
        raise ToolError("paper_id 不能为空")

    result = paper_text(
        paper_id,
        page=args.get("page"),
        section=args.get("section"),
        offset=int(args.get("offset") or 0),
        limit=int(args.get("limit") or 12),
    )
    if result.get("error"):
        raise ToolError(result["error"])
    return result


def _ask(args: dict) -> dict:
    from .rag import answer

    question = str(args.get("question") or "").strip()
    if not question:
        raise ToolError("question 不能为空")

    filters: dict[str, Any] = {}
    if args.get("paper_ids"):
        filters["paper_ids"] = list(args["paper_ids"])

    result = answer(
        question,
        limit=args.get("limit"),
        filters=filters,
        history=args.get("history"),
    )
    if result.error:
        raise ToolError(result.error)
    return result.to_dict()


def _list_papers(args: dict) -> dict:
    from .papers import list_papers, paper_dict

    rows, next_cursor, total = list_papers(
        query=args.get("query"),
        year=args.get("year"),
        venue=args.get("venue"),
        reading_status=args.get("reading_status"),
        has_repo=args.get("has_repo"),
        sort=str(args.get("sort") or "added"),
        limit=int(args.get("limit") or 20),
        cursor=args.get("cursor"),
    )
    return {
        "total": total,
        "count": len(rows),
        "next_cursor": next_cursor,
        "papers": [paper_dict(p) for p in rows],
    }


def _get_paper(args: dict) -> dict:
    from .papers import get_paper, paper_dict

    paper_id = str(args.get("paper_id") or "").strip()
    if not paper_id:
        raise ToolError("paper_id 不能为空")
    paper = get_paper(paper_id)
    if paper is None:
        raise ToolError("论文不存在")
    return paper_dict(paper, detail=True)


def _get_note(args: dict) -> dict:
    from .notes import get_note, list_notes, to_dict

    note_id = str(args.get("note_id") or "").strip()
    if note_id:
        note = get_note(note_id)
        if note is None:
            raise ToolError("笔记不存在")
        return to_dict(note)

    paper_id = str(args.get("paper_id") or "").strip()
    if not paper_id:
        raise ToolError("note_id 与 paper_id 至少要给一个")
    rows, _, _ = list_notes(paper_id=paper_id, limit=5)
    if not rows:
        return {"count": 0, "notes": [], "hint": "这篇论文还没有笔记。"}
    return {
        "count": len(rows),
        "notes": [to_dict(note) for note in rows],
    }


def _list_notes(args: dict) -> dict:
    from .notes import list_notes, to_dict

    rows, next_cursor, total = list_notes(
        query=args.get("query"),
        kind=args.get("kind"),
        status=args.get("status"),
        standalone=bool(args.get("standalone", False)),
        limit=int(args.get("limit") or 20),
        cursor=args.get("cursor"),
    )
    return {
        "total": total,
        "count": len(rows),
        "next_cursor": next_cursor,
        # 列表页不带正文——一次拉 20 篇正文会把上下文撑爆。
        # 要看正文用 get_note。
        "notes": [to_dict(note, include_content=False) for note in rows],
    }


def _list_tags(_args: dict) -> dict:
    from .papers import tag_facets

    return {"dimensions": tag_facets()}


def _get_code(args: dict) -> dict:
    from ..extensions import db
    from ..models import CodeRepo

    paper_id = str(args.get("paper_id") or "").strip()
    if not paper_id:
        raise ToolError("paper_id 不能为空")

    repos = db.session.query(CodeRepo).filter(CodeRepo.paper_id == paper_id).all()
    if not repos:
        return {
            "count": 0,
            "repos": [],
            "hint": "这篇论文没有关联的开源代码仓库。",
        }
    return {
        "count": len(repos),
        "repos": [
            {
                "name": repo.name,
                "url": repo.url,
                "local_path": repo.local_path,
                "vcs": repo.vcs,
                "head_commit": repo.head_commit,
                "language_stats": repo.language_stats or {},
                "readme_excerpt": (repo.readme_excerpt or "")[:2000],
            }
            for repo in repos
        ],
    }


def _stats(_args: dict) -> dict:
    from .indexer import index_stats
    from .papers import stats as paper_stats

    return {"papers": paper_stats(), "index": index_stats()}


# --------------------------------------------------------------------------
# 工具表
# --------------------------------------------------------------------------

TOOLS: tuple[AgentTool, ...] = (
    AgentTool(
        name="search",
        title="检索知识库",
        description=(
            "在论文原文分块、笔记正文里做混合检索（全文 + 中文 + 标题 + 向量融合）。"
            "**这是入口工具**：任何关于「这个知识库里有什么」的问题都从这里开始。\n\n"
            "返回的每条结果都带 `locator`（形如 `§3 Method p.4`）和 `chunk_id`，"
            "`locator` 可以直接作为出处写进你的回答。\n\n"
            "检索是字面/语义匹配，**不做推理**——"
            "问「这两篇方法有什么不同」这类需要理解的问题，先 search 拿到材料，"
            "或者直接用 ask。"
        ),
        cost=(
            "检索本身不走模型，但服务端**默认开启查询扩展**（把中文提问翻成英文术语），"
            "那一步会调用一次轻量模型，实测约 1.6 秒、几百 token。"
            "同一个查询词有缓存，重复查不再花钱。想省就少发几次、每次问得具体些——"
            "缩小 limit 并不能省掉这一步。"
        ),
        properties={
            "query": {"type": "string", "description": "查询词。中英文都行：中文提问也能命中英文论文。"},
            "limit": {"type": "integer", "description": "返回条数，默认 8。", "minimum": 1, "maximum": 50},
            "group_by_paper": {
                "type": "boolean",
                "description": "true 时同一篇论文最多出一条（只留最相关的）。默认 false。",
            },
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "限定只在这几篇论文里检索，便于追问某一篇。",
            },
            "tag_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "按标签过滤，多个标签是 AND 语义。",
            },
        },
        required=["query"],
        handler=_search,
    ),
    AgentTool(
        name="read_paper",
        title="按页/小节读论文原文",
        description=(
            "读取一篇论文的正文分块。**search 之后用它看上下文。**\n\n"
            "三种用法：\n"
            "  · 不传 page/section——返回**全篇目录**加开头几块，先看清结构；\n"
            "  · 传 `section`——按小节名子串匹配。**把 search 结果里的 "
            "`section_path` 原样填进来**（不要自己从 locator 里切字符串）；\n"
            "  · 传 `page`——取该页的正文，用来核对某句话是否真在第 N 页"
            "（search 结果里给了 `page_from`）。\n\n"
            "每块都带 `locator`，可直接引用。返回里的 `next_offset` 用来往下翻。"
        ),
        properties={
            "paper_id": {"type": "string", "description": "论文 id（search 结果里的 paper_id）。"},
            "section": {"type": "string", "description": "小节名，子串匹配，如 \"Method\"。"},
            "page": {"type": "integer", "description": "页码，1 起。取与该页有交集的块。"},
            "offset": {"type": "integer", "description": "从第几块开始，默认 0。"},
            "limit": {"type": "integer", "description": "返回块数，默认 12，上限 50。"},
        },
        required=["paper_id"],
        handler=_read_paper,
    ),
    AgentTool(
        name="ask",
        title="基于知识库问答（带引用校验）",
        description=(
            "检索 + 让大模型基于检索结果作答，返回带引用的回答。\n\n"
            "**代价与延迟都显著高于 search**，而且会花钱。适合「需要综合多篇论文"
            "直接给一个答案」的场景；只想拿材料自己判断时用 search。\n\n"
            "回答里的 `[1] [2]` 与 `citations[].marker` 对应，每条 citation 有 `check` 字段：\n"
            "  · `verified`——引文能在被引原文里逐字找到，可以放心用；\n"
            "  · `unverified`——模型没给出可核对的引文；\n"
            "  · `mismatched`——引文与原文对不上，**不要引用**。\n"
            "`grounded` 为 false 表示至少有一条 mismatched。\n\n"
            "知识库里没有答案时模型可能自行联网，结果放在**单独的 `web_citations`**，"
            "那里**没有校验**，与 `citations` 不是一个可信度，不要混用。"
        ),
        properties={
            "question": {"type": "string", "description": "问题。中英文都行。"},
            "limit": {"type": "integer", "description": "检索片段数，默认取服务端配置。"},
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "限定只在这几篇论文里检索。",
            },
            "history": {
                "type": "array",
                "items": {"type": "object"},
                "description": "多轮上下文 [{\"role\":\"user\",\"content\":\"...\"}]，最多用最近 6 轮。",
            },
        },
        required=["question"],
        handler=_ask,
        cost=(
            "调用大模型，比 search 贵一到两个数量级，也慢得多（含思考通常十几秒）。"
            "一次提问会顺带做一次检索，所以它也包含 search 的那笔开销。"
        ),
    ),
    AgentTool(
        name="list_papers",
        title="浏览论文列表",
        description=(
            "按条件翻论文列表，返回元数据（不含正文）。适合先看库里有什么，"
            "或者按年份/会议/阅读状态筛。\n\n"
            "关键词过滤走的是标题/作者/摘要的模糊匹配，**不是全文检索**——"
            "要找内容用 search。"
        ),
        properties={
            "query": {"type": "string", "description": "标题/作者/摘要关键词。"},
            "year": {"type": "integer", "description": "发表年份。"},
            "venue": {"type": "string", "description": "会议或期刊名。"},
            "reading_status": {
                "type": "string",
                "enum": ["unread", "reading", "read"],
                "description": "阅读状态。",
            },
            "has_repo": {"type": "boolean", "description": "只看有开源代码的。"},
            "sort": {
                "type": "string",
                "enum": ["added", "added_asc", "updated", "title", "year", "year_asc"],
                "description": "排序，默认 added（最新入库在前）。",
            },
            "limit": {"type": "integer", "description": "条数，默认 20。"},
            "cursor": {"type": "string", "description": "翻页游标，用上一次返回的 next_cursor。"},
        },
        handler=_list_papers,
    ),
    AgentTool(
        name="get_paper",
        title="取单篇论文详情",
        description=(
            "按 id 取一篇论文的完整元数据：摘要、作者、年份、会议、标签、"
            "关联的开源代码仓库、已有的笔记列表。**不含正文**——正文用 read_paper。"
        ),
        properties={
            "paper_id": {"type": "string", "description": "论文 id。"},
        },
        required=["paper_id"],
        handler=_get_paper,
    ),
    AgentTool(
        name="get_note",
        title="取笔记正文",
        description=(
            "取 AI 生成的精读笔记（中文 Markdown，含方法、实验、局限、复现要点等小节）。\n\n"
            "**这是本知识库里唯一的「已经消化过的中文内容」**——论文原文是英文的，"
            "想让中文问题得到中文表述，笔记往往比原文更好用。\n\n"
            "给 `note_id` 取某一篇；或给 `paper_id` 取这篇论文的全部笔记。"
        ),
        properties={
            "note_id": {"type": "string", "description": "笔记 id。与 paper_id 二选一。"},
            "paper_id": {"type": "string", "description": "论文 id，取它的全部笔记。"},
        },
        handler=_get_note,
    ),
    AgentTool(
        name="list_notes",
        title="浏览笔记列表",
        description=(
            "翻笔记列表，返回标题/标签/状态等元数据，**不含正文**（用 get_note 取）。"
            "`standalone=true` 只看不属于任何论文的独立笔记。"
        ),
        properties={
            "query": {"type": "string", "description": "标题关键词。"},
            "kind": {"type": "string", "description": "笔记类型，如 summary / manual。"},
            "status": {"type": "string", "description": "状态，如 draft / reviewed。"},
            "standalone": {"type": "boolean", "description": "只看独立笔记（不属于任何论文）。"},
            "limit": {"type": "integer", "description": "条数，默认 20。"},
            "cursor": {"type": "string", "description": "翻页游标。"},
        },
        handler=_list_notes,
    ),
    AgentTool(
        name="list_tags",
        title="列出标签词表",
        description=(
            "按维度列出全部标签及每篇论文/笔记的计数。用在你想按标签过滤之前——"
            "拿到确切的 tag_id 再传给 search 的 tag_ids。"
        ),
        properties={},
        handler=_list_tags,
    ),
    AgentTool(
        name="get_code",
        title="取论文的关联代码仓库",
        description=(
            "取一篇论文关联的开源仓库：仓库地址、本地克隆路径、主语言构成、README 摘要。\n\n"
            "本知识库只做「关联 + 浅索引」，**不能读仓库里的源码文件**——"
            "`local_path` 是给调用方自己去看的，本工具不返回文件内容。"
        ),
        properties={
            "paper_id": {"type": "string", "description": "论文 id。"},
        },
        required=["paper_id"],
        handler=_get_code,
    ),
    AgentTool(
        name="stats",
        title="知识库总览",
        description="论文数、笔记数、分块数、索引状态。用来判断这个库值不值得问，或者了解覆盖范围。",
        properties={},
        handler=_stats,
    ),
)


_BY_NAME: dict[str, AgentTool] = {tool.name: tool for tool in TOOLS}


def tool_specs() -> list[dict[str, Any]]:
    """``tools/list`` 的结果。

    **顺序固定**（按 TOOLS 的定义顺序），不按字典序也不按任何随机的顺序。
    新版 MCP 规范明确要求确定性顺序，好让客户端缓存；而且稳定顺序意味着
    模型每次看到的清单一致，不会因为顺序变化改变选择。
    """
    return [tool.to_spec() for tool in TOOLS]


def get_tool(name: str) -> AgentTool | None:
    return _BY_NAME.get(name)


def call(name: str, arguments: dict | None = None) -> Any:
    """执行一个工具。

    失败抛 ``ToolError``（调用方给错了参数）或原样抛出底层异常。
    调用方（MCP 层）负责把它转成 ``isError: true`` 的结果——按 MCP 规范，
    工具执行失败是**结果**而不是协议错误，得让模型看到失败原因。
    """
    tool = _BY_NAME.get(name)
    if tool is None:
        raise ToolError(f"没有名为 {name!r} 的工具。可用：{', '.join(sorted(_BY_NAME))}")
    return tool.handler(arguments or {})


__all__ = [
    "TOOLS",
    "AgentTool",
    "ToolError",
    "call",
    "get_tool",
    "tool_specs",
]
