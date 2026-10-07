"""对外接口的统一响应封装。

所有 ``/api/v1`` 的响应都长这样::

    成功  {"ok": true,  "data": ..., "meta": {...}}
    失败  {"ok": false, "error": {"code": "...", "message": "...", "details": {}}, "meta": {...}}

为什么要固定信封而不是直接返回裸数据：调用方多半是别的 agent（或按
OpenAPI 生成的客户端），它们需要**一致的成功/失败判别方式**。若一部分
接口成功返回数组、失败返回 ``{"error": ...}``，调用方就得为每个端点
单独写判断，这类代码几乎必然出错。

``meta`` 里固定带 ``request_id``，出问题时用户报这个 id 就能在日志/审计里定位。
"""

from __future__ import annotations

import base64
import binascii
import time
import uuid
from typing import Any

from flask import jsonify, request

# 分页上限。没有上限的话，一个 ?limit=1000000 就能把内存打满。
MAX_LIMIT = 200
DEFAULT_LIMIT = 50


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def _meta(extra: dict | None = None, started: float | None = None) -> dict:
    meta: dict[str, Any] = {
        "request_id": getattr(request, "kb_request_id", None) or new_request_id(),
    }
    if started is not None:
        meta["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    if extra:
        meta.update(extra)
    return meta


def ok(data: Any = None, *, status: int = 200, meta: dict | None = None, started: float | None = None):
    """成功响应。"""
    return jsonify({"ok": True, "data": data, "meta": _meta(meta, started)}), status


def error_response(
    code: str,
    message: str,
    status: int = 400,
    details: dict | None = None,
):
    """失败响应。

    ``code`` 是给机器看的稳定标识（``not_found`` / ``invalid_argument``…），
    ``message`` 是给人看的。调用方应该判断 code，而不是去匹配 message 文本。
    """
    return (
        jsonify(
            {
                "ok": False,
                "error": {"code": code, "message": message, "details": details or {}},
                "meta": _meta(),
            }
        ),
        status,
    )


# --------------------------------------------------------------------------
# 游标分页
# --------------------------------------------------------------------------


def encode_cursor(value: str) -> str:
    """把「上一页最后一条」的标识编码成游标。

    用不透明游标而不是 offset：offset 分页在数据变动时会漏条或重复，
    而知识库正在被后台任务持续写入，翻页时数据几乎一定在变。
    游标锚定在具体记录上，天然不受影响。
    """
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> str | None:
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        return base64.urlsafe_b64decode(padded.encode()).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None


def parse_paging() -> tuple[int, str | None]:
    """从查询串里取出 (limit, cursor)，并做边界钳制。"""
    raw_limit = request.args.get("limit", type=int) or DEFAULT_LIMIT
    limit = max(1, min(raw_limit, MAX_LIMIT))
    return limit, decode_cursor(request.args.get("cursor"))


def paged(data: list, *, next_cursor: str | None, limit: int, total: int | None = None):
    """分页响应。``next_cursor`` 为 None 表示没有下一页。

    **传进来的是「上一页最后一条的标识」这种原始值，编码由这里做。**
    不要把编码推给调用方：线格式是信封层的职责，散到各个端点就会漏。

    这不是假设——修之前 ``encode_cursor`` 从头到尾**没有任何调用者**，
    三个分页端点（papers / notes / jobs）都直接把裸 ID 塞进 ``next_cursor``。
    而 ``parse_paging`` 会把它当 base64 解，裸 ULID 解出来不是合法 UTF-8，
    于是 ``decode_cursor`` 返回 None、分页条件被静默跳过——**第二页永远
    等于第一页**，客户端按文档「原样回传游标」就会拿到同一批结果转到天荒地老。
    实测确认：两页返回的 id 完全相同，且 ``next_cursor`` 原样不变。
    """
    if next_cursor:
        next_cursor = encode_cursor(next_cursor)
    meta: dict[str, Any] = {"next_cursor": next_cursor, "limit": limit}
    if total is not None:
        meta["total"] = total
    return ok(data, meta=meta)


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "decode_cursor",
    "encode_cursor",
    "error_response",
    "new_request_id",
    "ok",
    "paged",
    "parse_paging",
]
