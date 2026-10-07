"""对外接口的自我描述：OpenAPI 规范 + 给 agent 的自述文档。

**为什么值得单独做。** 一套接口如果只能靠读源码才能调用，那它对外就是
不可用的——调用方是人还是 agent 都一样，agent 更甚：它没有「问一下同事」
这个选项，只能靠文档和错误信息摸索。

规范里的**路径是从 Flask 的 url_map 自动生成的**，不是手抄的。手抄的清单
一定会过期：新加了端点忘了补文档，调用方看不到；删了端点文档还留着，
调用方照着调会 404。自动生成后，规范里的端点集合永远等于真实存在的集合，
人工只需要补充「这个端点干什么用」。

没有描述的人工补充的端点会带上 ``x-undocumented: true``，这样「文档没写」
本身是可见的，而不是看起来像「这个端点没有参数」。
"""

from __future__ import annotations

import logging
from typing import Any

from flask import current_app, jsonify, request

from . import api_bp

log = logging.getLogger(__name__)

# 关键端点的说明。不在这里的端点依然会出现在规范里（路径自动生成），
# 只是没有摘要——见上面 x-undocumented 的说明。
ENDPOINT_DOCS: dict[str, dict[str, Any]] = {
    "api.ask_endpoint": {
        "summary": "基于知识库提问，返回带引用的回答",
        "description": (
            "检索知识库并生成回答。回答里的 `[1]`、`[2]` 是引用标记，"
            "与 citations 数组的 marker 一一对应。\n\n"
            "每个 citation 带 `check` 三态：`verified`（引文能在被引分块中"
            "逐字找到）、`unverified`（模型没给可核对的引文）、"
            "`mismatched`（引文对不上，最需要人工复核）。\n\n"
            "知识库里没有答案时，模型会自己决定是否联网检索；"
            "联网来源单独放在 `web_citations`，**没有校验**，"
            "与 `citations` 的可信度不是一个量级，不要混用。"
        ),
        "body": {
            "question": "问题（也可以用 q）",
            "limit": "检索片段数，默认取设置里的 retrieval.top_k",
            "paper_ids": "限定只在这几篇论文里检索",
            "history": "多轮上下文 [{role, content}]，最多用最近 6 轮",
        },
        "example": {"question": "DDPM 为什么用 ε-prediction？", "limit": 6},
    },
    "api.ask_stream_endpoint": {
        "summary": "同上，但以 SSE 流式返回",
        "description": (
            "事件类型：`sources`（先给检索结果，不必等模型开始输出）、"
            "`thinking`、`text`、`done`（含成型的 citations）、`error`。"
            "注意这是 GET 也能调，方便直接用 EventSource。"
        ),
    },
    "api.search_endpoint": {
        "summary": "混合检索，返回带页码定位的片段",
        "body": {"q": "查询词（也接受 query）", "limit": "条数"},
        "example": {"q": "world model autonomous driving", "limit": 5},
    },
    "api.list_papers_endpoint": {
        "summary": "论文列表，支持关键词/年份/会议/状态/标签过滤",
    },
    "api.get_paper_endpoint": {"summary": "单篇论文的元数据"},
    "api.get_paper_text": {
        "summary": "取论文正文分块（按页/按小节）",
        "description": (
            "「检索 → 读原文 → 引用」里的第二步。不传参数返回全篇目录加开头几块；"
            "`section` 按小节名子串匹配；`page` 取该页正文。"
            "每块带 `locator`（形如 `§3 Method p.4`），可直接作为出处。"
        ),
    },
    "api.get_paper_file": {"summary": "下载 PDF 原文（支持 Range 断点续传）"},
    "api.list_notes_endpoint": {"summary": "笔记列表"},
    "api.create_note_endpoint": {"summary": "新建笔记"},
    "api.update_note_endpoint": {"summary": "更新笔记（带乐观锁 version）"},
    "api.list_tags_endpoint": {"summary": "标签词表"},
    "api.health": {"summary": "健康检查：数据库、检索、模型配置状态"},
    "api.list_jobs": {"summary": "后台任务列表"},
    "api.job_events": {"summary": "任务的实时日志（SSE）"},
}

# 可选的过滤参数说明，用在多个列表端点上
COMMON_FILTERS = (
    "通用过滤参数：`q`（关键词）、`year`、`venue`、`reading_status`、"
    "`ingest_status`、`has_code`、`tag`（可重复，AND 语义）"
)


def _spec_paths() -> dict[str, Any]:
    """从 url_map 生成路径，再用 ENDPOINT_DOCS 补充说明。"""
    paths: dict[str, Any] = {}
    for rule in current_app.url_map.iter_rules():
        if not rule.rule.startswith("/api/v1/"):
            continue
        if rule.endpoint == "static":
            continue
        # MCP 说的是 JSON-RPC，不是这套 {ok,data,meta} 信封的 REST 规矩。
        # 把它收进 OpenAPI 会让符合规范的客户端以为能按 REST 调，
        # 而它只认 tools/list 那一套——两个协议混在一份规范里，两边都不准。
        if rule.endpoint.startswith("api.mcp_"):
            continue

        # Flask 的 <param> 转成 OpenAPI 的 {param}
        path = rule.rule
        for arg in rule.arguments:
            path = path.replace(f"<{arg}>", f"{{{arg}}}")
            path = path.replace(f"<int:{arg}>", f"{{{arg}}}")
            path = path.replace(f"<path:{arg}>", f"{{{arg}}}")

        methods = sorted(rule.methods - {"HEAD", "OPTIONS"})
        doc = ENDPOINT_DOCS.get(rule.endpoint, {})

        operation: dict[str, Any] = {
            "operationId": rule.endpoint.replace(".", "_"),
            "responses": {
                "200": {"description": "成功。响应体为 {ok:true, data, meta} 信封"},
                "401": {"description": "缺少或无效的 API Key"},
                "403": {"description": "Key 权限不足"},
            },
            "security": [{"bearerAuth": []}],
        }
        if doc.get("summary"):
            operation["summary"] = doc["summary"]
        if doc.get("description"):
            operation["description"] = doc["description"]
        if not doc:
            # 没人工写说明的端点也列出来，但显式标出「未文档化」，
            # 而不是让它看起来像「这个端点没有参数」
            operation["x-undocumented"] = True
            operation["summary"] = f"（未补充说明）{rule.endpoint}"

        params = [
            {"name": arg, "in": "path", "required": True, "schema": {"type": "string"}}
            for arg in rule.arguments
        ]
        if "GET" in methods:
            params.append({"name": "limit", "in": "query", "schema": {"type": "integer"}})
            params.append({"name": "cursor", "in": "query", "schema": {"type": "string"}})
        if params:
            operation["parameters"] = params

        if doc.get("body") or doc.get("example"):
            properties = {
                key: {"type": "string", "description": value}
                for key, value in (doc.get("body") or {}).items()
            }
            operation["requestBody"] = {
                "content": {
                    "application/json": {
                        "schema": {"type": "object", "properties": properties},
                        **({"example": doc["example"]} if doc.get("example") else {}),
                    }
                }
            }

        paths.setdefault(path, {}).update({m.lower(): operation for m in methods})
    return paths


@api_bp.get("/openapi.json")
def openapi_spec():
    """OpenAPI 3.1 规范。

    **不需要鉴权**：调用方得先能发现接口，才谈得上带 Key 调用。
    规范里不含任何机密，公开它不增加攻击面。
    """
    spec = {
        "openapi": "3.1.0",
        "info": {
            "title": "论文知识库 API",
            "version": current_app.config.get("KB_VERSION", "0.1.0"),
            "description": (
                "个人论文知识库的对外接口。\n\n"
                "**鉴权**：`Authorization: Bearer kb_xxx`。Key 在网页的设置页"
                "或 `flask kb key create` 创建，只显示一次。\n\n"
                "**响应信封**：成功 `{ok:true, data, meta}`，"
                "失败 `{ok:false, error:{code,message,details}, meta}`。"
                "每个响应都带 `X-Request-Id`，报问题时附上它。\n\n"
                "**分页**：列表端点用 `limit` + 不透明游标 `cursor`，"
                "下一页游标在 `meta.next_cursor`。\n\n"
                f"{COMMON_FILTERS}"
            ),
        },
        "servers": [{"url": request.url_root.rstrip("/")}],
        "components": {
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer"}
            }
        },
        "paths": _spec_paths(),
    }
    return jsonify(spec)


@api_bp.get("/agent-guide")
def agent_guide():
    """给 LLM agent 读的自述文档（纯文本）。

    刻意用纯文本而不是 JSON：这份东西是**给模型读的**，
    塞进上下文时不需要额外解析，token 也花得少。
    """
    settings = current_app.extensions["kb_settings"]
    base = request.url_root.rstrip("/")
    text = f"""# 论文知识库 · 调用指南

你可以用这套接口检索一个私人论文知识库（当前 {_paper_count()} 篇论文），
并基于原文回答带出处的问题。

## 鉴权
所有 /api/v1 下的接口（openapi.json 与本文除外）都需要：
    Authorization: Bearer <API_KEY>

## 如果你支持 MCP，优先用 MCP
本服务同时提供 MCP 接入面，工具清单与语义都是现成的，不必自己拼 HTTP：
    stdio:  python scripts/mcp_server.py       （客户端自己拉起，无需 Web 服务）
    HTTP:   POST {base}/api/v1/mcp            （JSON-RPC 2.0，需上面那个 Key）
十个工具全部只读：search / read_paper / get_note / list_papers / list_notes /
get_paper / get_code / list_tags / stats / ask。除了 ask 都不花钱。
下面这份 REST 说明是给「不支持 MCP」或需要写数据的场景用的。

## 核心调用范式：search → read → cite

1. **先检索**，拿到片段和它们的精确定位：
       POST {base}/api/v1/search
       {{"q": "world model autonomous driving", "limit": 8}}
   返回的每条形如：
       {{"chunk_id": "...", "paper_id": "...", "paper_title": "...",
         "locator": "§3 Method p.4", "text": "..."}}
   `locator` 已经拼好，可以直接写进你的回答里充当出处。

2. **需要更多上下文时**按论文取全文或按笔记取内容：
       GET  {base}/api/v1/papers/<paper_id>
       GET  {base}/api/v1/papers/<paper_id>/text
       GET  {base}/api/v1/notes?paper_id=<paper_id>

3. **直接问答**（内部会做检索 + 生成 + 引文核对）：
       POST {base}/api/v1/ask
       {{"question": "...", "limit": 6}}

## 关于引用：务必看 check 字段
回答里的 [1] [2] 与 citations 的 marker 对应。每条 citation 有 check：
    verified    引文能在被引分块里逐字找到 —— 可以放心引用
    unverified  模型没给出可核对的引文 —— 引用前自己核对原文
    mismatched  引文对不上 —— **不要引用**，多半是模型编的

`grounded` 为 false 时说明至少有一条 mismatched。

## 联网检索
知识库里没有答案时，模型会**自己决定**是否上网查（学术文献 / GitHub 仓库 /
通用网页）。联网结果放在**单独的 `web_citations` 字段**里，与 `citations` 分开：

    citations       来自知识库，带 check 三态，可逐字核对
    web_citations   来自互联网，**没有校验**，marker 形如 W1、W2

**两者的可信度不是一个量级**，请不要混用、也不要给 web_citations 加上
「已验证」之类的说法。每条 web_citation 带 url / kind(paper|code|web) /
source / published，可以自己点开核对。

联网开关与各来源的配置在设置页（`websearch.*`）。通用网页搜索需要配置
Tavily API Key，未配置时学术与代码检索仍然可用。

## 文件与任务
    GET  {base}/api/v1/papers/<paper_id>/file     PDF 原文（支持 Range）
    POST {base}/api/v1/papers                    上传/按 arXiv ID 入库（202 + job_id）
    GET  {base}/api/v1/jobs/<job_id>              查任务状态
    GET  {base}/api/v1/jobs/<job_id>/events       任务日志（SSE）

## 出错时
    {{"ok": false, "error": {{"code": "...", "message": "...", "details": {{}}}}}}
错误码：unauthorized / forbidden / not_found / invalid_argument /
        rate_limited / llm_error / conflict

## 完整规范
    GET {base}/api/v1/openapi.json

## 当前检索配置
    片段数默认 {settings.get("retrieval.top_k")}
    分组（同一篇论文只出一条）：{settings.get("retrieval.group_by_paper")}
"""
    return current_app.response_class(text, mimetype="text/plain; charset=utf-8")


def _paper_count() -> int:
    from ..extensions import db
    from ..models import Paper

    try:
        return db.session.query(Paper).filter(Paper.deleted_at.is_(None)).count()
    except Exception:
        return 0


__all__ = ["agent_guide", "openapi_spec"]
