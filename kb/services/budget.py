"""模型花费账本与预算闸门。

**为什么需要它。** 这套系统里花钱的地方很多：精读整篇论文、打标签、查询扩展、
联网检索前的判断 …… 而这些调用散在十几个地方。实测一次全库重新生成跑掉
132 万 token，事前估不出来、事后查不到 —— 「这个月花了多少、花在哪了」
完全无从回答。

只记 **token 数**，不记金额：token 是可核对的事实，价格是会变的配置。
一条记着「$0.23」的旧记录，在换了模型、调了价之后就没有意义了；
而「输入 120 万 token」永远是准的。折算成钱放在读取侧做。

**预算超限抛异常，不静默降级。** 静默降级会让「结果变差」看起来像
「模型不行」，排查方向会整个歪掉。宁可让任务明确失败，也不要让它悄悄
跑成一个更差的版本。
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from datetime import timedelta
from typing import Any

from ..extensions import db
from ..models import LLMUsage
from ..models.base import utcnow
from .llm.base import LLMError

log = logging.getLogger(__name__)


class BudgetExceeded(LLMError):
    """超出模型花费上限。消息面向用户。

    继承 ``LLMError`` 是为了让已有的错误处理（CLI 打印、流水线标记失败）
    不用改就能接住它；单独成类是为了让**重试逻辑能认出它并放弃重试**——
    超预算时重试只会继续撞墙，而且每撞一次都更接近账单上限。
    """


# 当前这笔调用属于哪个环节。用 contextvar 而不是参数层层传递——
# 调用链很深（pipeline → provider → complete），中间夹着通用函数，
# 为记账给每一层加参数会把签名污染得很厉害。
_ACTIVE: contextvars.ContextVar[tuple[str, str | None]] = contextvars.ContextVar(
    "kb_llm_kind", default=("other", None)
)


def current_kind() -> tuple[str, str | None]:
    return _ACTIVE.get()


@contextlib.contextmanager
def track(kind: str, *, ref: str | None = None):
    """把这一段里发生的模型调用记到指定环节名下。

    用法::

        with budget.track("read", ref=paper.id):
            result = provider.extract(...)
    """
    token = _ACTIVE.set((kind, ref))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


# --------------------------------------------------------------------------
# 记账
# --------------------------------------------------------------------------


def record(
    *,
    kind: str | None = None,
    model: str = "",
    ref: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    blocked: bool = False,
) -> None:
    """记一笔用量。

    **绝不抛异常。** 记账失败不该让一次已经成功的模型调用变成失败——
    用户拿到的是账单少一行，而不是答案丢了。
    """
    try:
        active_kind, active_ref = _ACTIVE.get()
        db.session.add(
            LLMUsage(
                kind=kind or active_kind,
                model=model or None,
                ref=ref or active_ref,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                cache_read_tokens=int(cache_read_tokens or 0),
                cache_write_tokens=int(cache_write_tokens or 0),
                blocked=blocked,
            )
        )
        db.session.commit()
    except Exception:
        log.debug("记录模型用量失败（不影响调用本身）", exc_info=True)


def record_usage(usage, *, model: str = "") -> None:
    """记下 Provider 返回的 ``Usage``。

    Provider 层唯一的记账入口：一次 API 调用记一条。工具循环里的每一轮
    都是独立的 API 调用，会各自经过这里——这正是我们想要的，
    因为每一轮都真的花了钱。
    """
    record(
        model=model,
        input_tokens=getattr(usage, "input_tokens", 0),
        output_tokens=getattr(usage, "output_tokens", 0),
        cache_read_tokens=getattr(usage, "cache_read_tokens", 0),
        cache_write_tokens=getattr(usage, "cache_write_tokens", 0),
    )


# --------------------------------------------------------------------------
# 闸门
# --------------------------------------------------------------------------


# 闸门看的是**滚动 24 小时**，不是自然日。
#
# 自然日会在午夜清零，于是「23:59 跑一个全库重读」能合法地把额度用两遍；
# 而且自然日要选时区，选错了用户会觉得账目对不上。滚动窗口没有这两个问题，
# 代价只是「今天花了多少」这句话要说成「过去 24 小时花了多少」。
WINDOW_HOURS = 24


def _setting(name: str, default: float) -> float:
    from flask import current_app

    with contextlib.suppress(Exception):
        settings = current_app.extensions.get("kb_settings")
        if settings is not None:
            value = settings.get(name)
            if value is not None:
                return float(value)
    return default


def _price() -> tuple[float, float]:
    """(输入单价, 输出单价)，单位：美元 / 百万 token。都为 0 表示不计价。"""
    return (
        _setting("llm.price_input_per_mtok", 0.0),
        _setting("llm.price_output_per_mtok", 0.0),
    )


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = _price()
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


def _window_start():
    return utcnow() - timedelta(hours=WINDOW_HOURS)


def _sum_tokens(*, since=None) -> tuple[int, int]:
    """(输入, 输出) token 合计。"""
    query = db.session.query(
        db.func.coalesce(db.func.sum(LLMUsage.input_tokens), 0),
        db.func.coalesce(db.func.sum(LLMUsage.output_tokens), 0),
    ).filter(LLMUsage.blocked.is_(False))
    if since is not None:
        query = query.filter(LLMUsage.created_at >= since)
    row = query.one()
    return int(row[0] or 0), int(row[1] or 0)


def spent_usd() -> float:
    """窗口内的花费（美元）。没配价格时恒为 0。"""
    if not any(_price()):
        return 0.0
    return estimate_cost(*_sum_tokens(since=_window_start()))


def check() -> None:
    """花钱之前先过这道闸门。超限抛 ``BudgetExceeded``。

    **两道闸，任意一道超了就停，且优先看 token。**
    token 闸不需要任何价格配置就能生效——正因为如此它才是默认开着的那道。
    金额闸只在用户主动填了单价之后才有意义（价格会变，我们不去猜）。

    限制的是**窗口内已经花掉的**量，不是这次调用要花多少：
    精确预估要同时知道提示词长度、输出长度和思考量，而输出长度本来就只能猜。
    用「已经超了就停」把超支控制在**一次调用**之内，比假装能精算更诚实。

    另外这道闸**不区分调用方**：后台批处理和网页问答共用一个额度。
    这是有意的——超额通常意味着某个循环失控了，这时候该停下来看看，
    而不是让网页问答继续把额度吃光。
    """
    limit_tokens = _setting("llm.budget_tokens", 0.0)
    limit_usd = _setting("llm.budget_usd", 0.0)
    window_in, window_out = _sum_tokens(since=_window_start())

    blocked_reason: str | None = None

    if limit_tokens > 0:
        used = window_in + window_out
        if used >= limit_tokens:
            blocked_reason = (
                f"模型用量已达上限（过去 {WINDOW_HOURS} 小时用了 {used:,} token，"
                f"上限 {int(limit_tokens):,}）。调高「用量上限」，"
                "或把它设为 0 关闭限制。"
            )

    if blocked_reason is None and limit_usd > 0 and any(_price()):
        used_usd = estimate_cost(window_in, window_out)
        if used_usd >= limit_usd:
            blocked_reason = (
                f"模型花费已达上限（过去 {WINDOW_HOURS} 小时约 ${used_usd:.2f}，"
                f"上限 ${limit_usd:.2f}）。调高「花费上限」，或把它设为 0 关闭限制。"
            )

    if blocked_reason is not None:
        # 被拦下的调用也记一条。**账本要能回答「为什么这批任务只跑了一半」**——
        # 只记花掉的钱，会看到一条戛然而止的曲线，看不出是闸门拦的。
        record(blocked=True)
        raise BudgetExceeded(blocked_reason)


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def summary(*, days: int | None = None) -> dict[str, Any]:
    """账本汇总：总量、按环节拆分、窗口用量与剩余额度。

    ``days`` 留空时统计**全部历史**——这是「我一共花了多少」的那个问题，
    和闸门看的 24 小时窗口是两回事，所以两个数都要给出来。
    """
    query = db.session.query(LLMUsage).filter(LLMUsage.blocked.is_(False))
    if days:
        query = query.filter(LLMUsage.created_at >= utcnow() - timedelta(days=days))

    rows = query.all()
    total_in = sum(r.input_tokens or 0 for r in rows)
    total_out = sum(r.output_tokens or 0 for r in rows)
    cache_read = sum(r.cache_read_tokens or 0 for r in rows)

    by_kind: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_kind.setdefault(row.kind, {"input": 0, "output": 0, "calls": 0})
        bucket["input"] += row.input_tokens or 0
        bucket["output"] += row.output_tokens or 0
        bucket["calls"] += 1

    window_in, window_out = _sum_tokens(since=_window_start())
    limit_tokens = int(_setting("llm.budget_tokens", 0.0))
    limit_usd = _setting("llm.budget_usd", 0.0)
    window_tokens = window_in + window_out

    return {
        "calls": len(rows),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cache_read_tokens": cache_read,
        "total_tokens": total_in + total_out,
        "cost_usd": estimate_cost(total_in, total_out),
        "priced": any(_price()),
        # --- 闸门视角：滚动窗口 ---
        "window_hours": WINDOW_HOURS,
        "window_tokens": window_tokens,
        "window_cost_usd": estimate_cost(window_in, window_out),
        "limit_tokens": limit_tokens,
        "limit_usd": limit_usd,
        "remaining_tokens": max(0, limit_tokens - window_tokens) if limit_tokens else None,
        "blocked": (
            (limit_tokens > 0 and window_tokens >= limit_tokens)
            or (limit_usd > 0 and any(_price()) and estimate_cost(window_in, window_out) >= limit_usd)
        ),
        # 被拦次数也按**同一个窗口**统计。之前这里数的是全部历史，
        # 而它显示在「滚动 24 小时」标题底下——用户会以为「刚被拦了 54 次」，
        # 实际是几周前累计的。两个口径混在一起，数字就没法用来判断现状。
        "blocked_calls": db.session.query(LLMUsage)
        .filter(LLMUsage.blocked.is_(True), LLMUsage.created_at >= _window_start())
        .count(),
        "blocked_calls_total": db.session.query(LLMUsage)
        .filter(LLMUsage.blocked.is_(True))
        .count(),
        "by_kind": by_kind,
    }


def top_consumers(*, limit: int = 10) -> list[dict[str, Any]]:
    """花费最多的关联对象（论文/会话）。回答「哪几篇最贵」。"""
    rows = (
        db.session.query(
            LLMUsage.ref,
            db.func.sum(LLMUsage.input_tokens + LLMUsage.output_tokens).label("tokens"),
        )
        .filter(LLMUsage.ref.isnot(None), LLMUsage.blocked.is_(False))
        .group_by(LLMUsage.ref)
        .order_by(db.text("tokens DESC"))
        .limit(limit)
        .all()
    )
    return [{"ref": ref, "tokens": int(tokens or 0)} for ref, tokens in rows]


__all__ = [
    "BudgetExceeded",
    "check",
    "current_kind",
    "estimate_cost",
    "record",
    "record_usage",
    "spent_usd",
    "summary",
    "top_consumers",
    "track",
]
