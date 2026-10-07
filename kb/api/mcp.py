"""MCP over HTTP（Streamable HTTP 的简化形态）。

给拉不起子进程的客户端用：请求体是 JSON-RPC 2.0，``POST /api/v1/mcp``。
带 ``Accept: text/event-stream`` 就回 SSE，否则回一条裸 JSON。

**协议实现只有一份**，在 ``kb/mcp.py``——这里只做传输层：认证（走本蓝图统一的
Bearer Key）、取 body、把结果按 HTTP 的习惯写回去。把协议判断也在这儿再写一遍
是典型的「两套实现迟早不一致」。

**为什么挂在 /api/v1 下而不是单独开一个口子。** 这样它自动继承这套接口已有的
鉴权、限流、审计和请求 ID；另起一个蓝图就要把这些再配一遍，而且很容易漏掉鉴权。
代价是它不能被当作普通 REST 端点收进 OpenAPI——见 ``docs._spec_paths`` 里的排除。
"""

from __future__ import annotations

import json
import logging

from flask import Response, jsonify, request

from ..mcp import Session, handle_message
from . import api_bp
from .auth import require_scope

log = logging.getLogger(__name__)

# 请求体上限。MCP 的请求都很小（工具名 + 参数），给 1MB 已经非常宽松——
# 不设上限的话，一个超大 body 能把内存吃满。
_MAX_BODY_BYTES = 1 << 20


def _sse(payload: dict) -> Response:
    """按 SSE 回一条就关。

    每个响应恰好一个 ``message`` 事件：这个实现里没有需要推送多次的
    请求（没有服务端主动通知），所以不需要保活连接。
    """
    body = f"event: message\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
    return Response(body, mimetype="text/event-stream", headers={"Cache-Control": "no-store"})


def _wants_sse() -> bool:
    return "text/event-stream" in (request.headers.get("Accept") or "")


def _respond(payload: dict | None, status: int = 200) -> Response:
    if payload is None:
        # 通知不回内容。202 是「收到了，但没什么可给你」。
        return Response("", status=202)
    if _wants_sse():
        return _sse(payload)
    return jsonify(payload), status


@api_bp.post("/mcp")
@require_scope("read")
def mcp_endpoint():
    """JSON-RPC 2.0 入口。

    **不需要先调 initialize。** 2026-07-28 版协议是无状态的，每个请求自带
    版本号与客户端能力；旧客户端发来的 ``initialize`` 也照常应答，之后它
    继续用旧格式请求即可（协议层按连接记这一点，HTTP 这边每个请求新建
    一个 session，所以旧客户端每次都要重新握手——这是无状态化的必然结果，
    客户端重发即可，代价很小）。
    """
    if request.content_length and request.content_length > _MAX_BODY_BYTES:
        return jsonify(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "请求体过大"},
            }
        ), 413

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "请求体不是合法 JSON"},
            }
        ), 400

    # HTTP 传输要求头里的版本与 _meta 里的版本一致，不一致必须报错而不是
    # 挑一个用——两边不一致说明客户端自己就有 bug，猜一边会让问题更难查。
    header_version = request.headers.get("MCP-Protocol-Version")
    messages = payload if isinstance(payload, list) else [payload]
    if header_version:
        for message in messages:
            if not isinstance(message, dict):
                continue
            meta = (message.get("params") or {}).get("_meta") or {}
            body_version = meta.get("io.modelcontextprotocol/protocolVersion")
            if body_version and body_version != header_version:
                return jsonify(
                    {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {
                            "code": -32020,
                            "message": (
                                "MCP-Protocol-Version 头与请求体 _meta 里的版本不一致"
                            ),
                            "data": {
                                "header": header_version,
                                "body": body_version,
                            },
                        },
                    }
                ), 400

    session = Session()
    if isinstance(payload, list):
        # 批量请求。空数组按 JSON-RPC 规范是无效请求。
        if not payload:
            return jsonify(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "空的批量请求"},
                }
            ), 400
        responses = [
            handle_message(message, session)
            for message in payload
            if isinstance(message, dict)
        ]
        responses = [r for r in responses if r is not None]
        if not responses:
            return _respond(None)
        return _respond(responses)

    if not isinstance(payload, dict):
        return jsonify(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "请求必须是对象或数组"},
            }
        ), 400

    return _respond(handle_message(payload, session))


@api_bp.get("/mcp")
def mcp_probe():
    """``GET`` 用来让调用方确认这个地址确实是个 MCP 端点。

    规范里的 Streamable HTTP 用 GET 开服务端推送流；本实现没有主动推送，
    所以用它做一次「你在不在」的自检，返回能力清单。
    """
    from ..mcp import PROTOCOL_VERSION, SUPPORTED_VERSIONS

    return jsonify(
        {
            "service": "kb-mcp",
            "protocolVersion": PROTOCOL_VERSION,
            "supportedVersions": list(SUPPORTED_VERSIONS),
            "transport": "streamable-http",
            "hint": "POST JSON-RPC 2.0 到本地址。stdio 方式见 README 的 MCP 一节。",
        }
    )


__all__ = ["mcp_endpoint", "mcp_probe"]
