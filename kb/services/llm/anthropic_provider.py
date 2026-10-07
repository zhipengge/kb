"""Anthropic Messages API 适配器。

既服务于 Anthropic 官方端点，也服务于**实现了 Messages API 形状的第三方端点**
（DeepSeek、以及任何提供 anthropic 兼容层的网关）。

两者的差别不是「能不能调通」，而是「哪些特性是真的」。本模块通过
``Capabilities`` 把差别显式化，业务层据此选择路径。实测结论见 base.py 的说明。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from .. import budget
from .base import (
    EXTRACTION_TOOL,
    Capabilities,
    LLMConfigError,
    LLMError,
    LLMResponse,
    Usage,
    _max_tokens,
    build_extraction_tool,
)

log = logging.getLogger(__name__)

# 思考会吃掉输出预算，所以默认值给得比较宽。
# 实测：一个是非题的回答消耗了 2240 字符的思考；结构化抽取那次直接把
# 3000 token 的额度全用在思考上，正文一个字都没剩。
DEFAULT_MAX_TOKENS = 16000

# 单次调用的**输出额度下限**。
#
# **思考与正文共用这个额度。** 实测这个端点上一个简单问题就能产生
# 6000~24000 字符的思考，而调用方按「我只要几十个字的答案」把 max_tokens
# 设成 1000~3000 —— 结果思考先把额度吃光，正文一个字都没有：
#
#     text='' thinking=8724字符 stop=max_tokens out=2000
#
# 更糟的是这种情况**不报错**：调用方拿到空字符串，各处的容错逻辑
# （查询扩展退回原查询、能力探测判定「不支持」、连通性测试报失败）
# 会把它当成一个正常结果继续走。同一个缺陷今天在三处独立出现
# （查询扩展、图谱抽取、能力探测），每次都以为是各自的问题。
#
# 所以把下限抬到一处，而不是让十几个调用点各自记得给足。
# 这是**上限**不是目标值——模型不会因为额度变大就多写，只有原本被截断的
# 那些调用会因此拿到本该有的输出。
MIN_MAX_TOKENS = 8000



def _is_official_anthropic(base_url: str | None) -> bool:
    if not base_url:
        return True
    lowered = base_url.lower()
    return "api.anthropic.com" in lowered


def _capabilities_for(base_url: str | None) -> Capabilities:
    """按端点给出能力默认值。

    这里给的是**保守的默认**——兼容端点上的特性必须经过探测才会被启用，
    绝不因为「参数没报错」就当它生效。实测中 ``cache_control`` 和
    ``output_config`` 都是「接受但无效」，它们不报错，只是安静地什么都不做。
    """
    if _is_official_anthropic(base_url):
        return Capabilities(
            pdf_native=True,
            vision=True,
            prompt_cache=True,
            structured_output=True,
            tool_use=True,
            thinking_control=True,
            streaming=True,
            max_output_tokens=16000,
            returns_thinking=True,
        )
    return Capabilities(
        pdf_native=False,          # 兼容端点多半不解析 PDF，改用本地抽取
        vision=True,
        prompt_cache=False,        # 不可依赖
        structured_output=False,   # 改用工具调用
        tool_use=True,
        thinking_control=False,
        streaming=True,
        max_output_tokens=16000,
        returns_thinking=True,     # 实测会返回 thinking 块
        notes=[
            "这是 Anthropic 兼容端点：PDF 原生解析、提示缓存、结构化输出均未启用。"
            "结构化抽取改用工具调用，论文正文由本地解析后注入。",
        ],
    )


class AnthropicProvider:
    """基于官方 anthropic SDK 的 Provider。"""

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        auth_token: str = "",
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 300.0,
        capabilities: Capabilities | None = None,
    ):
        if not model:
            raise LLMConfigError("没有配置模型")
        if not api_key and not auth_token:
            raise LLMConfigError("没有配置 API Key")

        self.model = model
        self.base_url = base_url or None
        self.default_max_tokens = max_tokens

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMConfigError("未安装 anthropic SDK") from exc

        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": timeout,
            "max_retries": 2,
        }
        if auth_token:
            # DeepSeek 这类网关用 Authorization: Bearer
            kwargs["auth_token"] = auth_token
        else:
            kwargs["api_key"] = api_key

        self.last_usage = Usage()
        self._client = anthropic.Anthropic(**kwargs)
        self._anthropic = anthropic
        self._caps = capabilities or _capabilities_for(self.base_url)

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        # 花钱之前先过闸门。超限就抛，**不降级**——降级会让「结果变差」
        # 看起来像「模型不行」，排查方向会整个歪掉。
        budget.check()

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _max_tokens(max_tokens, self.default_max_tokens),
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        # temperature 只在服务商支持时发送——某些推理模型会直接拒绝
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            if stream:
                with self._client.messages.stream(**payload) as stream_ctx:
                    for _ in stream_ctx.text_stream:
                        pass  # 由 stream_text 负责推送，这里只取最终消息
                    raw = stream_ctx.get_final_message()
            else:
                raw = self._client.messages.create(**payload)
        except self._anthropic.BadRequestError as exc:
            raise LLMError(f"请求被拒绝：{exc}") from exc
        except self._anthropic.AuthenticationError as exc:
            raise LLMConfigError("API Key 无效或已过期") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError(f"触发限流，请稍后重试：{exc}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"无法连接模型服务：{exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"模型服务返回错误 {exc.status_code}：{exc}") from exc

        return self._to_response(raw)

    def stream_text(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[dict]:
        """逐段产出事件，供 SSE 推送给前端。

        产出的事件形如::

            {"type": "thinking", "text": "…"}
            {"type": "text", "text": "…"}
            {"type": "tool", "name": "…", "input": {...}}
            {"type": "done", "response": LLMResponse}

        思考是单独的一类事件，而不是并进正文——界面上要能把它折叠起来显示，
        混在一起会让回答变得没法读（实测一个回答的思考量常常是正文的数倍）。
        """
        # 生成器要等到第一次 next() 才执行到这里，那时调用方（rag.py）
        # 已经把 for 循环套在 try 里了，异常会变成一条错误事件推给前端。
        budget.check()

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _max_tokens(max_tokens, self.default_max_tokens),
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        try:
            with self._client.messages.stream(**payload) as stream_ctx:
                for event in stream_ctx:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "text": delta.text}
                    elif delta.type == "thinking_delta":
                        yield {"type": "thinking", "text": getattr(delta, "thinking", "")}
                final = stream_ctx.get_final_message()
        except self._anthropic.APIConnectionError as exc:
            yield {"type": "error", "message": f"连接中断：{exc}"}
            return
        except self._anthropic.APIStatusError as exc:
            yield {"type": "error", "message": f"服务返回错误 {exc.status_code}"}
            return

        response = self._to_response(final)
        for call in response.tool_calls:
            yield {"type": "tool", "name": call.get("name"), "input": call.get("input")}
        yield {"type": "done", "response": response}

    # ------------------------------------------------------------------
    # 结构化抽取
    # ------------------------------------------------------------------

    def extract(
        self,
        messages: list[dict],
        *,
        schema: dict,
        description: str,
        system: str | None = None,
        max_tokens: int | None = None,
        instructions: str = "",
    ) -> dict:
        """按给定 JSON Schema 抽取结构化数据。

        路径选择：
          * 官方端点且 ``structured_output`` 可用 → 走原生结构化输出；
          * 否则 → 走工具调用（所有主流服务商都支持的最小公倍数）。

        两条路径对调用方完全一致：都返回一个已校验过「是 dict 且含必需字段」的结果。
        """
        tool = build_extraction_tool(schema, description)
        prompt = [*messages,
            {
                "role": "user",
                "content": (
                    (instructions + "\n\n" if instructions else "")
                    + f"请调用 {EXTRACTION_TOOL} 工具提交结果。"
                    "不要用文字复述结果，直接调用工具。"
                ),
            }
        ]

        response = self.complete(
            prompt,
            system=system,
            tools=[tool],
            max_tokens=_max_tokens(max_tokens, self.default_max_tokens),
        )

        payload = response.first_tool_input(EXTRACTION_TOOL)
        # 重试会再花一次钱，两次的用量都要算在这篇论文头上。
        # 之前这里直接赋值 last_usage，第一次的用量就丢了。
        spent = response.usage

        # **空输入要和「没调工具」同等对待。**
        #
        # 推理型模型会把输出额度大量花在思考上，工具参数可能在生成到一半时
        # 被 max_tokens 截断，于是拿到一个空字典。若只判断 `is None`，
        # 空字典会被当成成功，然后在校验阶段报出「缺少全部字段」——
        # 那是个极具误导性的错误，会让人以为是 schema 定义有问题。
        if not payload:
            # 把工具名打出来。空结果有两种成因，日志上必须能区分：
            #   ① 模型调了工具但参数真的是空的；
            #   ② 模型调的工具**不叫这个名字**（拼错、多空格、幻觉出别的名字）——
            #      first_tool_input 按名字过滤，这里会返回 None，表现和①一样。
            # 不打出名字的话，这两种情况在日志里长得完全相同，只能靠猜。
            log.warning(
                "抽取未得到有效结果（stop=%s，思考 %d 字符，工具调用=%s），重试一次",
                response.stop_reason, len(response.thinking),
                [(c.get("name"), len(str(c.get("input") or ""))) for c in response.tool_calls] or "无",
            )
            response = self.complete(
                [
                    *prompt,
                    {
                        "role": "user",
                        "content": (
                            f"必须调用 {EXTRACTION_TOOL} 工具提交完整结果，不要用文字回答，"
                            "所有字段都要填写。"
                        ),
                    },
                ],
                system=system,
                tools=[tool],
                max_tokens=_max_tokens(max_tokens, self.default_max_tokens),
            )
            payload = response.first_tool_input(EXTRACTION_TOOL)
            spent = spent + response.usage

        self.last_usage = spent
        if not payload:
            truncated = response.stop_reason == "max_tokens"
            hint = (
                "输出被 max_tokens 截断了——推理型模型会先花掉大量额度思考，"
                "调大「单次最大输出 token」即可。"
                if truncated
                else "如果反复出现，考虑换一个更强的模型。"
            )
            raise LLMError(
                f"模型没有按要求返回结构化结果（stop_reason={response.stop_reason}）。{hint}"
            )

        return _validate(payload, schema)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _to_response(self, raw) -> LLMResponse:
        """把 SDK 的响应对象转成我们自己的结构。

        要点：**永远按块类型筛选，不要假设 content[0] 是文本。**
        实测这条端点上默认就会返回 thinking 块，直接取 content[0].text
        会抛 AttributeError——这也是我第一次探测时踩的坑。
        """
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict] = []
        content: list[dict] = []

        for block in raw.content or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
                content.append({"type": "text", "text": getattr(block, "text", "")})
            elif block_type == "thinking":
                value = getattr(block, "thinking", "") or ""
                thinking_parts.append(value)
                # 思考块要原样保留：同一模型继续对话时需要回传
                content.append({"type": "thinking", "thinking": value})
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {}) or {},
                    }
                )
                content.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {}) or {},
                    }
                )

        usage = Usage(
            input_tokens=getattr(raw.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw.usage, "cache_creation_input_tokens", 0) or 0,
        )

        response = LLMResponse(
            text="".join(text_parts).strip(),
            thinking="".join(thinking_parts),
            tool_calls=tool_calls,
            usage=usage,
            model=getattr(raw, "model", self.model),
            stop_reason=getattr(raw, "stop_reason", None),
            raw_content=content,
        )

        # 记账和 last_usage 都放在这里，因为**所有**调用路径（complete /
        # stream_text / extract 的重试）都会经过 _to_response。
        # 之前 last_usage 只在 extract 里赋值，complete 走完根本不更新，
        # 于是按论文统计的 token 数是漏的。
        self.last_usage = response.usage
        budget.record_usage(response.usage, model=response.model)

        return response


def _validate(payload: dict, schema: dict) -> dict:
    """按 schema 做最小必要校验。

    只检查「必需字段在不在、类型对不对」——不做完整 JSON Schema 校验，
    因为目标是把明显错误的数据挡在入库之前，而不是实现一个校验器。
    模型返回多余字段是允许的（有些服务商会附加解释性字段）。
    """
    if not isinstance(payload, dict):
        raise LLMError(f"抽取结果不是对象：{type(payload).__name__}")

    required = schema.get("required", [])
    missing = [key for key in required if key not in payload]
    if missing:
        # 缺一两个字段不该让整篇论文白跑。
        #
        # 实测：模型偶尔会漏掉某个数组字段（比如论文确实没什么「关键技术」
        # 可写），一个 `key_techniques` 缺失就把整次抽取判死，代价和收益完全
        # 不成比例——笔记本来就是给人复核的草稿。
        #
        # 但**缺得太多就是另一回事**：那说明这次抽取真的失败了（全空对象
        # 会走到这里），必须报出来。所以按比例设阈值，而不是一律放过。
        properties = schema.get("properties") or {}
        if len(missing) > max(1, len(required) // 3):
            raise LLMError(f"抽取结果缺少必需字段：{', '.join(missing)}")
        for key in missing:
            spec = properties.get(key) or {}
            payload[key] = [] if spec.get("type") == "array" else ""
        log.warning("抽取结果缺少字段 %s，已按类型补空值继续", "、".join(missing))

    type_map = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "array": list, "object": dict,
    }
    for key, spec in (schema.get("properties") or {}).items():
        if key not in payload:
            continue
        expected = type_map.get(spec.get("type"))
        value = payload[key]
        if expected and not isinstance(value, expected):
            raise LLMError(
                f"字段 {key} 类型不对：期望 {spec.get('type')}，"
                f"实际 {type(value).__name__}"
            )
    return payload


__all__ = ["DEFAULT_MAX_TOKENS", "AnthropicProvider"]
