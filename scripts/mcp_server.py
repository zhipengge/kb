#!/usr/bin/env python
"""MCP stdio 服务：把知识库接到任何 MCP 客户端上。

    claude mcp add kb -- /path/to/.venv/bin/python /path/to/scripts/mcp_server.py

**stdout 是协议通道，除 JSON-RPC 之外的任何东西都不能往那儿写。**
一行多余的日志就会让客户端解析失败，而且症状是「连上了但什么都不工作」，
很难查。所以这里：

  * 日志全部改道 stderr（客户端会把 stderr 当服务端调试输出，照常显示）；
  * ``print`` 被换成一个会报警的函数——真有人加了一行调试输出时，
    立刻能在日志里看到，而不是等客户端报一句看不懂的解析错误。

**不启动内嵌 worker。** Web 服务起来时会带一个后台任务线程（扫描、索引、
精读都在里面跑）。MCP 是「按需查询」的进程，可能同时被开好几个，
每个都拖一份任务循环去抢同一个任务队列，纯属添乱。

**应用只建一次、context 只推一次。** 建 Flask app 要读配置、连数据库、
建 FTS 表，几十毫秒起步。每次调用重建的话，一次对话里问十来个问题就白等
小半秒。这里在启动时建好并一直推着 app context。
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path

# 允许直接 `python scripts/mcp_server.py` 跑（客户端就是这么拉起来的），
# 这时项目根不在 sys.path 里。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# **必须在导入 kb 之前设**：create_app 会读它决定要不要起后台任务线程。
#
# 用直接赋值而不是 setdefault：`pipenv run` 会在启动 python 之前就把 .env
# 灌进环境，所以这里读到的多半已经是 1，setdefault 一个字都不会改。
# 这不是「偏好」而是这个进程的正确性要求——MCP 服务是按需查询的，
# 可能同时开好几个，每个都拖一份任务循环去抢同一个任务队列只会互相打架。
os.environ["KB_WORKER_EMBEDDED"] = "0"


def _configure_logging() -> None:
    """日志一律走 stderr，stdout 留给协议。"""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(os.environ.get("KB_LOG_LEVEL", "WARNING"))


def _guard_stdout() -> None:
    """把 print 换成报警，保住 stdout 的纯洁。"""

    def _forbidden(*args, **kwargs):
        logging.getLogger("kb.mcp").error(
            "有人在 MCP stdio 服务里调用了 print（参数 %r）。"
            "stdout 是协议通道，写别的东西会让客户端解析失败。"
            "要输出调试信息请用日志（它走 stderr）。",
            args[:1],
        )

    import builtins

    builtins.print = _forbidden  # type: ignore[assignment]


def main() -> int:
    _configure_logging()
    _guard_stdout()

    from kb import create_app
    from kb.mcp import Session, handle_message

    app = create_app()
    log = logging.getLogger("kb.mcp")

    # stdout 重新包一层 UTF-8：客户端管道在 Windows 上默认可能是 GBK，
    # 而返回内容几乎必然含中文。
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    reader = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")

    session = Session()
    log.info("MCP stdio 服务已就绪（%s）", app.config.get("KB_VERSION", "?"))

    # app context 推一次、全程复用。服务层的函数都要读 current_app/数据库会话。
    with app.app_context():
        for line in reader:
            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                # 解析不出来就回一条标准错误，别让整个进程退出——
                # 一行脏输入不该终结这次会话。
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"JSON 解析失败：{exc}"},
                }
            else:
                try:
                    response = handle_message(message, session)
                except Exception as exc:  # pragma: no cover - 兜底
                    log.exception("处理消息失败")
                    response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id") if isinstance(message, dict) else None,
                        "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
                    }

            if response is None:
                continue  # 通知，不回响应
            out.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
            out.flush()

            # 每条请求后清一次会话，避免一个请求里的异常状态污染下一个。
            # 不 rollback 的话，某次工具调用中间失败会让后续查询都撞上
            # 「此会话已失效」。
            try:
                from kb.extensions import db

                db.session.rollback()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
