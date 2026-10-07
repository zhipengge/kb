"""对外接口的鉴权与授权。

鉴权（你是谁）用 Bearer Token；授权（你能做什么）用 scope。

scope 分四级，粒度刻意粗：``read`` < ``write`` < ``ingest`` < ``admin``。
更细的粒度（按资源、按方法）在这里收益很低——调用方是用户自己配置的
agent，而不是互不信任的第三方。粗粒度让「给一个只读的 agent 发只读 Key」
这件事变得直观，而这正是实际需要区分的主要场景。
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

from flask import current_app, g, request

from ..models import ApiKey
from ..services import apikeys

log = logging.getLogger(__name__)

# scope 的包含关系：更大权限的 Key 自动拥有更小权限
_SCOPE_ORDER = {"read": 0, "write": 1, "ingest": 2, "admin": 3}


def current_key() -> ApiKey | None:
    return getattr(g, "kb_api_key", None)


def actor_name() -> str:
    key = current_key()
    if key is not None:
        return f"key:{key.name}"
    return "anonymous"


def _extract_token() -> str | None:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip()
    # 也接受 X-API-Key，方便那些不便设 Authorization 的客户端
    return request.headers.get("X-API-Key")


def has_scope(required: str) -> bool:
    key = current_key()
    if key is None:
        return False
    granted = set(key.scopes or [])
    if "admin" in granted:
        return True
    threshold = _SCOPE_ORDER.get(required, 99)
    return any(_SCOPE_ORDER.get(s, 99) >= threshold for s in granted)


def require_scope(scope: str) -> Callable:
    """装饰器：要求调用方具备某个 scope。"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = current_key()
            if key is None:
                from .envelope import error_response

                return error_response(
                    "unauthorized",
                    "缺少或无效的 API Key。请在 Authorization 头里带上 Bearer <key>。",
                    401,
                )
            if not has_scope(scope):
                from .envelope import error_response

                return error_response(
                    "forbidden",
                    f"当前 Key 没有 {scope} 权限（已有：{', '.join(key.scopes or []) or '无'}）",
                    403,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def authenticate_request() -> None:
    """在 before_request 里调用一次，把鉴权结果放进 g。"""
    mode = current_app.config.get("KB_AUTH_MODE", "apikey")
    g.kb_api_key = None
    g.kb_auth_mode = mode

    if mode == "none":
        # 完全可信的本机环境。仍然标注出来，让审计日志能区分这两种情况。
        return

    token = _extract_token()
    if not token:
        return

    key = apikeys.verify_key(token)
    if key is None:
        log.warning("无效的 API Key（前缀 %s…）来自 %s", token[:12], request.remote_addr)
        return

    g.kb_api_key = key
    try:
        apikeys.touch_key(key)
    except Exception:
        # 记账失败不该让请求失败
        log.exception("更新 Key 使用记录失败")


__all__ = [
    "actor_name",
    "authenticate_request",
    "current_key",
    "has_scope",
    "require_scope",
]
