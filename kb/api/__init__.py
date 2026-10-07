"""对外接口（``/api/v1``）。

这个蓝图是给**其他 agent** 用的：它必须比网页端更讲规矩——
稳定的响应信封、机器可读的错误码、游标分页、幂等键、可发现的 OpenAPI 文档。
网页端可以随便改版，接口不行。

每个请求都会写审计日志，因为通过接口进来的是自动化调用：
出问题时「谁在什么时候改了哪篇论文的标签」必须有据可查。
"""

from __future__ import annotations

import logging
import time

from flask import Blueprint, g, request

from . import auth
from .envelope import new_request_id

log = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


@api_bp.before_request
def _before():
    g.kb_request_id = new_request_id()
    g.kb_started = time.perf_counter()
    auth.authenticate_request()


@api_bp.after_request
def _after(response):
    response.headers["X-Request-Id"] = getattr(g, "kb_request_id", "")

    # 只审计有副作用的调用：把每次检索也记一遍会让审计表迅速淹没在噪音里，
    # 反而查不到真正重要的写操作。
    if (
        request.method in {"POST", "PUT", "PATCH", "DELETE"}
        and response.status_code < 500
        and not request.path.endswith("/ping")
    ):
        try:
            _write_audit(response)
        except Exception:
            log.exception("写审计日志失败")
    return response


def _write_audit(response) -> None:
    from flask import current_app

    if not current_app.extensions["kb_settings"].get("security.audit_enabled"):
        return

    from ..extensions import db
    from ..models import AuditLog

    db.session.add(
        AuditLog(
            actor=auth.actor_name(),
            action=f"{request.method.lower()} {request.endpoint or request.path}",
            target=request.path,
            method=request.method,
            path=request.full_path.rstrip("?"),
            status_code=response.status_code,
            ip=request.remote_addr,
            meta={"request_id": getattr(g, "kb_request_id", None)},
        )
    )
    db.session.commit()


# 路由模块（导入即注册）。放在文件末尾是因为它们要 `from . import api_bp`，
# 而 api_bp 必须已经定义好——这也是这里用 noqa 抑制 E402 的原因。
from . import chat, mcp, notes, papers, search, system, tags  # noqa: E402,F401

__all__ = ["api_bp"]

from . import docs  # noqa: E402,F401  — 导入即注册文档路由
