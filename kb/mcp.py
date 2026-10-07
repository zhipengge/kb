"""MCP（Model Context Protocol）接入面。

让任何支持 MCP 的客户端（Claude Code、Claude Desktop、各种 agent 框架）
直接查这个知识库，不需要写一行胶水代码。

**协议形状按 2026-07-28 版实现，同时兼容旧版握手。** 这一版有个大改动：
协议变成**无状态**的——``initialize`` 握手被删除，改由每个请求在
``params._meta`` 里自带 ``io.modelcontextprotocol/protocolVersion``，
并新增 ``server/discover`` 让服务端自报能力。旧的 ``initialize`` 流程
（2025-11-25 及更早）仍然兼容，因为现实里的客户端不会同时升级。

两条通路都必须留：只实现新版，老客户端连不上；只实现旧版，新客户端
拿不到 ``resultType`` 等必填字段。判断依据是**请求里有没有带版本的 _meta**，
不是客户端自称是谁——规范明确说了 ``clientInfo`` 不可信、不得据此改变行为。

**为什么手写而不装官方 SDK。** 官方 ``mcp`` 包会拖进 14 个新依赖，包括
starlette + uvicorn（第二套 Web 框架）、opentelemetry、jsonschema ——
而这个项目已经跑着 Flask、全部运行时依赖都是精挑过的。stdio 上真正需要的
协议面只有 ``initialize`` / ``server/discover`` / ``tools/list`` / ``tools/call``
（外加 ``ping``），下面一百多行就说清楚了，换来零新依赖。
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

SERVER_NAME = "kb"
SERVER_TITLE = "论文知识库"

# 本服务实现的最新协议版本。
PROTOCOL_VERSION = "2026-07-28"

# 兼容的版本清单。顺序 = 从新到旧，``server/discover`` 按这个顺序上报，
# 客户端从中挑一个双方都支持的。
SUPPORTED_VERSIONS: tuple[str, ...] = (
    "2026-07-28",
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)

# 无状态协议里，版本号藏在这个 _meta 键下面
_META_VERSION = "io.modelcontextprotocol/protocolVersion"
_META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
_META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

# 工具清单是静态的（编译进代码），可以放心让客户端长时间缓存。
# 一小时足够省掉重复的 tools/list，又不至于在升级后让客户端拿着过期清单太久。
_TOOLS_TTL_MS = 3_600_000

# JSON-RPC / MCP 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNSUPPORTED_PROTOCOL_VERSION = -32022


def server_version() -> str:
    from flask import current_app

    try:
        return str(current_app.config.get("KB_VERSION", "0.1.0"))
    except Exception:
        return "0.1.0"


def _server_info() -> dict[str, str]:
    return {"name": SERVER_NAME, "title": SERVER_TITLE, "version": server_version()}


def _instructions() -> str:
    """``server/discover`` 里给模型的说明。

    这段会被客户端塞进模型的系统上下文，所以**只说「怎么用」**，
    不复述工具清单——清单在 ``tools/list`` 里，重复一遍白花 token
    还有两份不一致的风险。
    """
    return (
        "这是一个私人论文知识库，可以检索论文原文与 AI 精读笔记，"
        "并给出带出处、且经过原文校验的回答。\n"
        "典型顺序：search 找材料 → read_paper / get_note 读上下文 → 引用时带上 locator。\n"
        "需要综合多篇直接成段回答时才用 ask（它会调用大模型，较慢且产生费用）。\n"
        "引用可信度看 citation 的 check 字段：mismatched 的不要引用。"
    )


class Session:
    """一条连接上的协议状态。

    **只记一件事：这条连接是不是走旧版握手。**
    无状态协议本不该有连接状态，但要在同一根管道上同时伺候新旧两代客户端，
    就必须记住「对面已经发了 initialize」，否则没法决定后续响应里要不要塞
    ``resultType`` 这类新版必填字段。除此之外不存任何东西——
    尤其不缓存客户端能力（规范明确禁止跨请求推断能力）。
    """

    __slots__ = ("legacy",)

    def __init__(self) -> None:
        self.legacy = False


def _result(payload: dict, *, session: Session, cacheable: bool = False) -> dict:
    """按当前协议版本补全结果信封。

    新版必须带 ``resultType``；列表类结果还必须带 ``cacheScope`` 与 ``ttlMs``。
    旧版客户端不认识这些字段，多给会怎样？规范允许结果里出现额外字段，
    但老客户端的解析未必宽容，所以旧版连接上**一个都不加**。
    """
    if session.legacy:
        return payload

    out = dict(payload)
    out["resultType"] = "complete"
    if cacheable:
        # 工具清单与能力声明对所有调用方都一样，不含任何私人数据，
        # 所以是 public——中间的缓存代理可以复用。
        out["cacheScope"] = "public"
        out["ttlMs"] = _TOOLS_TTL_MS
    out["_meta"] = {_META_SERVER_INFO: _server_info()}
    return out


def _error(msg_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def _ok(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _requested_version(message: dict) -> str | None:
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    value = meta.get(_META_VERSION)
    return value if isinstance(value, str) else None


def handle_message(message: dict, session: Session) -> dict | None:
    """处理一条 JSON-RPC 消息。

    返回响应字典；**通知（没有 id）返回 None**——通知不回响应，
    回了会让严格按规范实现的客户端报协议错误。
    """
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(
            message.get("id") if isinstance(message, dict) else None,
            INVALID_REQUEST,
            "不是合法的 JSON-RPC 2.0 消息",
        )

    method = message.get("method")
    msg_id = message.get("id")
    is_notification = "id" not in message

    if not isinstance(method, str):
        return None if is_notification else _error(msg_id, INVALID_REQUEST, "缺少 method")

    # ---- 版本协商 ----
    #
    # 带版本的请求按无状态协议处理，先验版本再干活。
    # 不带版本的是旧版客户端，走下面的 initialize 分支。
    requested = _requested_version(message)
    if requested is not None and requested not in SUPPORTED_VERSIONS:
        # 规范要求把「你要的是什么、我支持什么」都告诉客户端，让它自己重挑一个。
        return _error(
            msg_id,
            UNSUPPORTED_PROTOCOL_VERSION,
            f"不支持的协议版本 {requested}",
            {"requested": requested, "supported": list(SUPPORTED_VERSIONS)},
        )
    if requested is not None:
        session.legacy = False

    if method == "server/discover":
        return _ok(
            msg_id,
            _result(
                {
                    "supportedVersions": list(SUPPORTED_VERSIONS),
                    "capabilities": {"tools": {"listChanged": False}},
                    "instructions": _instructions(),
                },
                session=session,
                cacheable=True,
            ),
        )

    if method == "initialize":
        # 旧版握手。客户端报它支持的版本，双方取交集：
        # 能对上就回它那个版本（规范要求回同一个），对不上就回我们最新的，
        # 由客户端决定要不要继续。
        params = message.get("params") or {}
        client_version = params.get("protocolVersion")
        chosen = (
            client_version
            if client_version in SUPPORTED_VERSIONS
            else PROTOCOL_VERSION
        )
        session.legacy = True
        return _ok(
            msg_id,
            {
                "protocolVersion": chosen,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": _server_info(),
                "instructions": _instructions(),
            },
        )

    if method in ("notifications/initialized", "notifications/cancelled"):
        return None

    if method == "ping":
        # 2026-07-28 已删掉 ping（无状态了，不需要保活），
        # 但旧客户端会发，回个空结果比报「方法不存在」友好得多。
        return None if is_notification else _ok(msg_id, {})

    if method == "tools/list":
        from .services.agent_tools import tool_specs

        return _ok(
            msg_id,
            _result({"tools": tool_specs()}, session=session, cacheable=True),
        )

    if method == "tools/call":
        return _handle_tool_call(message, session)

    if is_notification:
        return None
    return _error(msg_id, METHOD_NOT_FOUND, f"不支持的方法 {method}")


def _handle_tool_call(message: dict, session: Session) -> dict | None:
    """执行工具并包成结果。

    **工具执行失败走 ``isError``，不是 JSON-RPC 错误。** 这是规范的要求，
    也是更有用的做法：参数写错、论文不存在，模型需要**看到**失败原因才能
    改一改再试；包成协议错误的话，很多客户端会直接中断这一轮。
    只有「没有这个工具」才回协议错误——那是调用方搞错了协议层面的事。
    """
    from .services.agent_tools import call, get_tool

    msg_id = message.get("id")
    params = message.get("params") or {}
    name = params.get("name")
    arguments = params.get("arguments") or {}

    if not isinstance(name, str) or not name:
        return _error(msg_id, INVALID_PARAMS, "tools/call 缺少 name")
    if not isinstance(arguments, dict):
        return _error(msg_id, INVALID_PARAMS, "arguments 必须是对象")
    if get_tool(name) is None:
        return _error(msg_id, METHOD_NOT_FOUND, f"没有名为 {name!r} 的工具")

    try:
        payload = call(name, arguments)
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        result = {"content": [{"type": "text", "text": text}], "isError": False}
    except Exception as exc:
        # 参数错误与内部错误对模型来说是同一件事：这条路走不通，换一条。
        # 但要把原因原样带上，别换成一句「执行失败」——那样模型只能瞎试。
        log.warning("MCP 工具 %s 执行失败：%s", name, exc, exc_info=True)
        result = {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "isError": True,
        }

    if session.legacy:
        return _ok(msg_id, result)
    return _ok(msg_id, {**result, "resultType": "complete", "_meta": {_META_SERVER_INFO: _server_info()}})


__all__ = [
    "PROTOCOL_VERSION",
    "SUPPORTED_VERSIONS",
    "Session",
    "handle_message",
]
