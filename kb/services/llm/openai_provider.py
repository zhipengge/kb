"""OpenAI 兼容接口适配器。

覆盖 DeepSeek、Qwen、Kimi、GLM、vLLM、Ollama、LM Studio 等——
它们都实现 ``/v1/chat/completions`` 这一形状。

结构化抽取同样走**工具调用**而不是 ``response_format``：后者在自建服务上
支持度参差不齐（vLLM 要看版本、Ollama 只支持 ``format: json`` 不支持 schema），
而工具调用是各家都有的。统一走一条路，出错时只需要排查一种情况。
"""

from __future__ import annotations

import json
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



class OpenAICompatProvider:
    """基于 openai SDK 的兼容适配器。"""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 300.0,
        capabilities: Capabilities | None = None,
    ):
        if not model:
            raise LLMConfigError("没有配置模型")
        if not api_key and not base_url:
            raise LLMConfigError("需要配置 API Key 或服务地址")

        self.model = model
        self.base_url = base_url or None
        self.default_max_tokens = max_tokens

        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise LLMConfigError("未安装 openai SDK") from exc

        # 本地推理服务（Ollama / vLLM）通常不校验 key，但 SDK 要求非空
        self.last_usage = Usage()
        self._client = openai.OpenAI(
            api_key=api_key or "not-needed",
            base_url=self.base_url,
            timeout=timeout,
            max_retries=2,
        )
        self._openai_errors = openai

        # 兼容端点的能力默认值同样保守：这些特性在自建服务上普遍不可用，
        # 与其猜，不如让业务层走稳妥路径。
        self._caps = capabilities or Capabilities(
            pdf_native=False,
            vision=True,
            prompt_cache=False,
            structured_output=False,
            tool_use=True,
            thinking_control=False,
            streaming=True,
            max_output_tokens=max_tokens,
            returns_thinking=False,
            notes=["OpenAI 兼容端点：结构化抽取走工具调用，PDF 由本地解析后注入文本。"],
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    # ------------------------------------------------------------------
    # 消息格式转换
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_messages(messages: list[dict], system: str | None) -> list[dict]:
        """把 Anthropic 形状的消息转成 OpenAI 形状。

        业务层统一用 Anthropic 的消息格式（它是超集：内容块列表能同时表达
        文本、图片、工具调用），转换集中在这里做。
        """
        converted: list[dict] = []
        if system:
            converted.append({"role": "system", "content": system})

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")

            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                continue

            # 把内容块列表拆成「文本 + 图片」的 OpenAI 多模态格式
            parts: list[dict] = []
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif block_type == "image":
                    source = block.get("source") or {}
                    if source.get("type") == "base64":
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{source.get('media_type')};base64,{source.get('data')}"
                                },
                            }
                        )
                elif block_type == "tool_result":
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        }
                    )
                elif block_type == "document":
                    # 兼容端点不解析 PDF，这里给出明确提示而不是静默丢弃
                    log.warning("OpenAI 兼容端点不支持 PDF 文档块，已跳过该块")
                # thinking / tool_use 块在转换方向（Anthropic->OpenAI）上不参与

            if parts:
                converted.append({"role": role, "content": parts})

        return converted

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]

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
        budget.check()  # 超限抛异常，不降级

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages, system),
            "max_tokens": _max_tokens(max_tokens, self.default_max_tokens),
        }
        if tools:
            payload["tools"] = self._to_openai_tools(tools)
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            raw = self._client.chat.completions.create(**payload)
        except self._openai_errors.BadRequestError as exc:
            raise LLMError(f"请求被拒绝：{exc}") from exc
        except self._openai_errors.AuthenticationError as exc:
            raise LLMConfigError("API Key 无效") from exc
        except self._openai_errors.RateLimitError as exc:
            raise LLMError(f"触发限流：{exc}") from exc
        except self._openai_errors.APIConnectionError as exc:
            raise LLMError(f"无法连接服务：{exc}") from exc
        except self._openai_errors.APIStatusError as exc:
            raise LLMError(f"服务返回错误 {exc.status_code}") from exc

        return self._to_response(raw)

    def stream_text(self, messages: list[dict], **kwargs: Any) -> Iterator[dict]:
        budget.check()  # 生成器首次迭代时才执行，调用方已在 try 内

        system = kwargs.get("system")
        tools = kwargs.get("tools")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages, system),
            "max_tokens": kwargs.get("max_tokens") or self.default_max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = self._to_openai_tools(tools)

        text_parts: list[str] = []
        try:
            stream = self._client.chat.completions.create(**payload)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    text_parts.append(delta.content)
                    yield {"type": "text", "text": delta.content}
        except self._openai_errors.APIConnectionError as exc:
            yield {"type": "error", "message": f"连接中断：{exc}"}
            return
        except self._openai_errors.APIStatusError as exc:
            yield {"type": "error", "message": f"服务返回错误 {exc.status_code}"}
            return

        yield {
            "type": "done",
            "response": LLMResponse(text="".join(text_parts).strip(), model=self.model),
        }

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
        tool = build_extraction_tool(schema, description)
        prompt = [*messages,
            {
                "role": "user",
                "content": (
                    (instructions + "\n\n" if instructions else "")
                    + f"请调用 {EXTRACTION_TOOL} 工具提交结果，不要用文字复述。"
                ),
            }
        ]

        response = self.complete(
            prompt, system=system, tools=[tool],
            max_tokens=_max_tokens(max_tokens, self.default_max_tokens),
        )
        payload = response.first_tool_input(EXTRACTION_TOOL)
        # last_usage 由 _to_response 统一维护，这里不再重复赋值
        if payload is None:
            raise LLMError(
                f"模型没有按要求返回结构化结果（finish_reason={response.stop_reason}）"
            )

        from .anthropic_provider import _validate

        return _validate(payload, schema)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _to_response(self, raw) -> LLMResponse:
        choice = raw.choices[0] if raw.choices else None
        if choice is None:
            return LLMResponse(model=self.model, error="服务未返回任何结果")

        message = choice.message
        tool_calls: list[dict] = []

        for call in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                log.warning("工具参数不是合法 JSON：%s", call.function.arguments)
                arguments = {}
            tool_calls.append(
                {"id": call.id, "name": call.function.name, "input": arguments}
            )

        usage_raw = getattr(raw, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_raw, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_raw, "completion_tokens", 0) or 0,
        )

        response = LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            usage=usage,
            model=getattr(raw, "model", self.model),
            stop_reason=getattr(choice, "finish_reason", None),
        )

        # 和 anthropic 适配器一样：所有调用路径都经过这里，
        # 记账和 last_usage 只在这一处更新。
        self.last_usage = response.usage
        budget.record_usage(response.usage, model=response.model)

        return response


__all__ = ["DEFAULT_MAX_TOKENS", "OpenAICompatProvider"]
