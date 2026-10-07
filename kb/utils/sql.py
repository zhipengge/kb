"""SQL 相关的安全小工具。

SQL 的参数绑定只能用于**值**，表名、列名这类**标识符**没法参数化——
它们必须拼进语句文本里。所以凡是要拼标识符的地方，都必须先经过这里的校验。

这个项目里所有被拼接的标识符都是内部生成的（常量表名、从配置派生的向量表名），
理论上不会是攻击载荷。但校验的价值不在于「防住当前的攻击」，
而在于**把「这个名字是可信的」从隐含假设变成显式断言**：
以后若有人让表名来自用户输入，这里会立刻抛错，而不是安静地多出一个注入点。
"""

from __future__ import annotations

import re

# SQLite 标识符：字母或下划线开头，之后是字母、数字、下划线。
# 故意比 SQLite 实际允许的范围更严——宽松的规则会放进引号、空格、
# 分号等一切能改变语句结构的东西。
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class UnsafeIdentifierError(ValueError):
    """标识符不合法。这是编程错误，不是用户错误。"""


def safe_identifier(name: str) -> str:
    """校验一个将被拼进 SQL 的标识符，合法则原样返回。

    不合法的标识符一律抛错而不是转义或丢弃：走到这里说明前面的代码
    对「这个名字从哪来」的判断出了问题，静默修正会把问题藏起来。
    """
    if not name or not _IDENTIFIER.match(name):
        raise UnsafeIdentifierError(
            f"不安全的 SQL 标识符：{name!r}。"
            "标识符只能由字母、数字、下划线组成，且不能以数字开头。"
        )
    return name


def quote_identifier(name: str) -> str:
    """校验并用双引号包裹标识符。

    包裹之后连保留字（order、group 之类）也能安全用作名字。
    """
    return f'"{safe_identifier(name)}"'


__all__ = ["UnsafeIdentifierError", "quote_identifier", "safe_identifier"]
