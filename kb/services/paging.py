"""游标翻页：让游标与排序基准保持一致。

**这里解决的是一类静默丢数据的问题。** 游标翻页的下一页条件是
「排在上一页最后一条之后」，所以游标必须锚定在**排序用的那个字段**上。
如果排序按 ``title``、而游标按 ``id`` 过滤，两套基准对不上：
一部分记录既不满足 ``id < cursor``、又排在已返回区间之后，于是**永远取不到**，
而客户端拿到的是一个看似完整、实则缺了几条的列表。

实测（85 篇论文 / 85 篇笔记）：按 ``updated`` 排序翻页，笔记只取到 81 条，
另外 4 条既没出现在任何一页、也没有任何报错。

**游标格式是 ``<排序键>|<id>``。** id 作为决胜字段（tiebreaker）：
排序键相同的记录靠 id 保证全序，否则同一秒创建的两条记录翻页时会互相顶掉。
没有 ``|`` 时按「纯 id」处理，兼容 ``sort=added`` 这类本来就按 id 排的情况。

**比较交给 SQLite 的类型亲和性。** 排序键序列化成字符串后直接参与比较：
DATE/TEXT 列按字典序（ISO 时间串的字典序恰好等于时间序），INTEGER 列由
SQLite 依列亲和性把字符串转回整数。这是 SQLite 特有的行为——换数据库时
这里要改成按列类型显式转换。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import and_, or_

CURSOR_SEP = "|"


def key_to_str(value: Any) -> str:
    """把排序键的值序列化进游标。"""
    if value is None:
        # 用哨兵而不是空串：空串会和「真的空字符串标题」混淆
        return "\x00null"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def key_from_str(raw: str) -> Any:
    """从游标里还原排序键。"""
    return None if raw == "\x00null" else raw


def make_cursor(key_value: Any, row, id_attr: str = "id") -> str:
    """由「本页最后一条」的排序键与 id 生成游标。"""
    return f"{key_to_str(key_value)}{CURSOR_SEP}{getattr(row, id_attr)}"


def split_cursor(cursor: str | None) -> tuple[str | None, str | None]:
    """拆成 ``(排序键, id)``。没有分隔符时整串当作 id。"""
    if not cursor:
        return None, None
    if CURSOR_SEP not in cursor:
        return None, cursor
    key, _, last_id = cursor.partition(CURSOR_SEP)
    return key, last_id or None


def cursor_condition(column, id_column, cursor: str | None, *, descending: bool, parse=None):
    """构造「排在游标之后」的 WHERE 条件。

    返回 None 表示不需要过滤（没有游标，或游标无法解析）。

    条件形如（降序）::

        column < key  OR  (column = key AND id < last_id)

    id 参与其中是为了应对**排序键相同**的记录：只比 column 的话，
    同一批次里剩下的那几条会被判为「不大于游标」而丢失。

    ``parse`` 把游标里的字符串还原成**列的真实类型**。这一步不能省：
    游标里存的是文本，而 DATE 列要的是 datetime——直接把字符串绑上去，
    SQLAlchemy 会按列类型转换，而 ``isoformat()`` 产出的是
    ``2026-10-07T01:11:49``（T 分隔），SQLite 里存的却是
    ``2026-10-07 01:11:49``（空格分隔）。两者字典序不同，比较恒假，
    表现是**翻页彻底不动**（降序）或**第二页就空**（升序）。
    实测就是这么坏的：改成文本比较后，desc 排序死循环、asc 排序只出一页。
    """
    if not cursor:
        return None

    key, last_id = split_cursor(cursor)
    if key is None:
        # 纯 id 游标（sort 本身按 id 排）
        return id_column < last_id if last_id else None

    value = parse(key_from_str(key)) if parse else key_from_str(key)
    if descending:
        return or_(
            column < value,
            and_(column == value, id_column < last_id),
        )
    return or_(
        column > value,
        and_(column == value, id_column > last_id),
    )


__all__ = [
    "CURSOR_SEP",
    "cursor_condition",
    "key_from_str",
    "key_to_str",
    "make_cursor",
    "split_cursor",
]
