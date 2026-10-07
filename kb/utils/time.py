"""时间相关的展示工具。"""

from __future__ import annotations

from datetime import UTC, datetime

# 中文的「几秒前 / 几分钟前」不区分单复数，比英文简单
_STEPS: tuple[tuple[int, str], ...] = (
    (60, "秒"),
    (60, "分钟"),
    (24, "小时"),
    (30, "天"),
    (12, "个月"),
)


def humanize_delta(value: datetime | None, *, now: datetime | None = None) -> str:
    """把时间点描述成「3 分钟前」这类相对时间。

    数据库里存的是带时区的 UTC 时间；这里统一先转成 aware 再比较，
    否则遇到 naive datetime（老数据、手工 SQL 插入）会直接抛
    "can't subtract offset-naive and offset-aware datetimes"。
    """
    if value is None:
        return "—"

    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)

    delta = reference - value
    seconds = delta.total_seconds()

    if seconds < 0:
        return "刚刚"
    if seconds < 10:
        return "刚刚"

    magnitude = seconds
    unit = "秒"
    for factor, next_unit in _STEPS:
        if magnitude < factor:
            break
        magnitude /= factor
        unit = next_unit
    else:
        return value.strftime("%Y-%m-%d")

    return f"{int(magnitude)} {unit}前"


def to_utc(value: datetime | None) -> datetime | None:
    """把任意 datetime 归一成带 UTC 时区的形式。"""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def iso(value: datetime | None) -> str | None:
    """序列化成 API 用的 ISO 8601 字符串。"""
    normalized = to_utc(value)
    return normalized.isoformat().replace("+00:00", "Z") if normalized else None


__all__ = ["humanize_delta", "iso", "to_utc"]
