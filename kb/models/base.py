"""模型基础件：ULID 主键、时间戳、JSON 列约定。"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# --------------------------------------------------------------------------
# ULID
# --------------------------------------------------------------------------

# Crockford Base32：去掉了容易混淆的 I/L/O/U
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id() -> str:
    """生成一个 ULID（26 字符，字典序 = 时间序）。

    为什么不用自增整数或 UUID：
      * 自增整数在导入/合并时容易打架，也暴露总量；
      * UUID4 完全随机，作为主键会让 B-tree 插入变成随机散布，
        而且按创建时间排序要额外读一个字段。
    ULID 前缀是 48 位毫秒时间戳，新记录总是插在索引尾部，且直接字符串排序
    就是时间序——列表页按时间倒序时不需要额外索引。

    自己实现是为了不引入依赖，实现本身只有十几行。
    """
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big")

    # 时间戳 48 位 -> 10 个 base32 字符；随机数 80 位 -> 16 个
    chars = []
    for shift in range(45, -1, -5):
        chars.append(_CROCKFORD[(ts_ms >> shift) & 0x1F])
    for shift in range(75, -1, -5):
        chars.append(_CROCKFORD[(rand >> shift) & 0x1F])
    return "".join(chars)


def utcnow() -> datetime:
    """带时区的当前时间。

    统一用 aware datetime：naive datetime 在跨时区（这台机器同时有 Windows 与
    WSL 两套本地时间）时会静默算错，而且是那种半年后才发现的错。
    """
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# 声明式基类
# --------------------------------------------------------------------------


class Base(DeclarativeBase):
    """所有模型的基类。

    SQLite 没有原生 JSON 类型，SQLAlchemy 的 JSON 会以 TEXT 存储并自动
    序列化/反序列化。对于 authors、aliases、meta 这类「结构不固定但不需
    要按内部字段检索」的数据正合适——真需要检索的字段一律提升为独立列。

    这里**不要**定义 ``type_annotation_map``：Flask-SQLAlchemy 会基于这个类
    再派生一层，而 SQLAlchemy 不允许「基类自带 registry 又带 type_annotation_map」。
    """



class TimestampMixin:
    """创建/更新时间。默认值放在数据库侧（server_default），
    这样直接用 SQL 插入（迁移脚本、批量导入）也能自动填充。"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        server_default=func.now(),
        nullable=False,
    )


class IdMixin:
    """ULID 主键。"""

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=new_id)


__all__ = ["Base", "IdMixin", "TimestampMixin", "new_id", "utcnow"]
