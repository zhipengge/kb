"""LLM 接入层的公共契约。

**这个模块的存在理由：不同服务商的能力差别很大，而差别会渗透到业务逻辑里。**

实测例子（DeepSeek 的 Anthropic 兼容端点）：
  * 接受 PDF 文档块，但不解析——直接返回「无法确定」；
  * 接受 ``output_config.format``，但输出被吞掉，只剩思考；
  * 接受 ``cache_control``，但 cache_read 恒为 0，是空操作；
  * 思考恒开且不可关，还会吃掉 ``max_tokens`` 的预算。

如果业务代码直接写死「用 output_config 拿 JSON」「把 PDF 塞给模型」，
换个服务商就会静默失效——不是报错，而是产出错误的结果，这更难发现。

所以业务层只说「我要这个结构的数据」，由 Provider 按自己的能力选择实现路径。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


# 单次调用的**输出额度下限**。
#
# **思考与正文共用这个额度，所以「少要一点」在这里不存在。** 这个端点上一个
# 简单问题就能产生 6000~24000 字符的思考；调用方按「我只要几十个字的答案」
# 把 max_tokens 设成 1000~3000，结果思考先把额度吃光，正文一个字都没有：
#
#     text='' thinking=8724字符 stop=max_tokens out=2000
#
# 更糟的是这种情况**不报错**：调用方拿到空字符串，各处的容错逻辑
# （查询扩展退回原查询、能力探测判定「不支持」、连通性测试报失败）
# 会把它当成一个正常结果继续走。同一个缺陷先后在三处独立出现
# （查询扩展、图谱抽取、能力探测），每次都以为是各自的问题。
#
# **下限最终定在默认值本身（16000），而不是折中的 8000。** 8000 那次也踩了
# 同一个坑：图谱抽取显式要了 8000，实测一篇笔记思考 14738 字符就把额度吃光，
# JSON 一个字符都没吐出来——**两次调用全废，笔记还没建成**。
#
# 推理就在数据里：思考量由模型决定，调用方无从预估。那么任何低于「一次完整
# 回答」的额度都是在赌，而赌输的代价（整次调用作废）远大于赌赢的收益
# （省下的额度本来也不会被用掉）。
#
# 这是**上限**不是目标值——模型不会因为额度变大就多写，只有原本被截断的
# 那些调用会因此拿到本该有的输出。
#
# **只在 base 里定义一份。** 两个 provider 里曾各有一个同名常量（还停在
# 8000），它们不被任何代码读到，却看上去像在生效——我自己就被它们骗过一次，
# 改完才发现改的是死代码。要调这个值，改这里，别处没有第二处。
MIN_MAX_TOKENS = 16000


def _max_tokens(requested: int | None, default: int) -> int:
    """算出这次调用实际给多少输出额度。见 MIN_MAX_TOKENS 的说明。"""
    return max(int(requested or default), MIN_MAX_TOKENS)


class LLMError(RuntimeError):
    """模型调用失败。消息面向用户。"""


class LLMConfigError(LLMError):
    """配置不完整或不可用（缺 key、地址错等）。"""


@dataclass
class Capabilities:
    """服务商支持什么、不支持什么。

    默认值刻意保守：不声明支持的一律当作不支持，业务层就会走稳妥的降级路径。
    """

    # 能否直接吃 PDF 文档块并真正解析内容
    pdf_native: bool = False
    # 能否理解图片
    vision: bool = True
    # cache_control 是否真的产生缓存命中
    prompt_cache: bool = False
    # output_config.format / response_format 是否可用
    structured_output: bool = False
    # 工具调用
    tool_use: bool = True
    # 能否控制思考的开关与深度
    thinking_control: bool = False
    # 流式输出
    streaming: bool = True

    # 单次输出的 token 上限。思考和正文共用这个额度，
    # 所以实际可用空间要按「思考占掉一大半」来估。
    max_output_tokens: int = 8192

    # 是否把思考内容一并返回（会影响消息历史的结构）
    returns_thinking: bool = False

    # 探测能力的来源与时间，供界面显示
    probed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pdf_native": self.pdf_native,
            "vision": self.vision,
            "prompt_cache": self.prompt_cache,
            "structured_output": self.structured_output,
            "tool_use": self.tool_use,
            "thinking_control": self.thinking_control,
            "streaming": self.streaming,
            "max_output_tokens": self.max_output_tokens,
            "returns_thinking": self.returns_thinking,
            "probed": self.probed,
            "notes": list(self.notes),
        }


@dataclass
class Usage:
    """一次调用的用量。用于界面上的成本显示与预算控制。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class LLMResponse:
    """一次调用的结果。"""

    text: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    stop_reason: str | None = None
    raw_content: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def first_tool_input(self, name: str | None = None) -> dict | None:
        """取第一个工具调用的参数（可指定工具名）。

        工具参数正常是对象，但**兼容端点偶尔会把它双重编码成 JSON 字符串**
        发回来。SDK 的解析是宽松的（``construct_type``：类型对不上就原样返回），
        所以这种响应里 ``input`` 是 ``str`` 而不是 ``dict``。

        这种响应里参数内容其实是完整的。早期版本只判断 ``isinstance(x, dict)``，
        于是症状表现为「模型调了工具，但参数是空的」——触发一次多半无用的重试，
        严重时整篇论文失败。这里补一道解析，并把命中情况写进日志：
        如果这条日志出现，就证实了双重编码确实在发生（而不是模型真的没填）。
        """
        for call in self.tool_calls:
            if name is not None and call.get("name") != name:
                continue
            payload = call.get("input")
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, str) and payload.strip():
                try:
                    parsed = json.loads(payload)
                except (ValueError, TypeError):
                    log.warning(
                        "工具 %s 的参数是字符串且无法解析为 JSON（%d 字符）：%r",
                        call.get("name"), len(payload), payload[:200],
                    )
                    continue
                if isinstance(parsed, dict):
                    log.warning(
                        "工具 %s 的参数被双重编码成 JSON 字符串，已解析（%d 字符）",
                        call.get("name"), len(payload),
                    )
                    return parsed
        return None


class LLMProvider(Protocol):
    """服务商适配器需要实现的接口。"""

    name: str
    model: str

    @property
    def capabilities(self) -> Capabilities: ...

    def complete(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> LLMResponse: ...

    def stream_text(self, messages: list[dict], **kwargs: Any): ...


# --------------------------------------------------------------------------
# 结构化抽取
# --------------------------------------------------------------------------


EXTRACTION_TOOL = "submit_result"


def build_extraction_tool(schema: dict, description: str) -> dict:
    """把 JSON Schema 包成一个工具定义。

    这是「结构化抽取」的通用实现方式。为什么不用各家的原生结构化输出：

      * Anthropic 原生支持 ``output_config.format``，但兼容端点未必；
      * OpenAI 用 ``response_format``，形状完全不同；
      * 而**工具调用是所有主流服务商都支持的最小公倍数**。

    实测 DeepSeek 的兼容端点：``output_config.format`` 会把输出整个吞掉
    （只剩思考、没有文本），而工具调用工作得很好。所以统一走工具路径，
    反而比追着各家特性跑更可靠。
    """
    return {
        "name": EXTRACTION_TOOL,
        "description": description,
        "input_schema": schema,
    }
