"""LLM 接入层。

对外只暴露两个函数：``get_provider()`` 拿一个可用的 Provider，
``chat_provider()`` / ``fast_provider()`` 拿按用途选好模型的 Provider。
业务代码不应该关心底层是哪家服务商。
"""

from __future__ import annotations

import logging

from .base import (
    EXTRACTION_TOOL,
    Capabilities,
    LLMConfigError,
    LLMError,
    LLMResponse,
    Usage,
    build_extraction_tool,
)

log = logging.getLogger(__name__)


def get_provider(*, fast: bool = False):
    """按当前设置构造 Provider。

    ``fast=True`` 时用轻量模型（打标、分类这类高频调用）。

    每次调用都新建实例：Provider 内部持有 HTTP 客户端，而设置随时可能在
    网页上被改（换模型、换地址）。缓存实例会让「改了设置但不生效」成为一个
    难以排查的问题——重建的开销完全可以忽略。
    """
    from flask import current_app

    settings = current_app.extensions["kb_settings"]

    provider_name = settings.get("llm.provider")
    base_url = (settings.get("llm.base_url") or "").strip() or None
    credential = settings.get_secret("llm.api_key")

    model = (
        settings.get("llm.fast_model") if fast else settings.get("llm.deep_model")
    ) or settings.get("llm.deep_model")

    if not model:
        raise LLMConfigError("没有配置模型。到「设置 → 模型」里填写。")

    # 凭据缺失时给出可操作的提示，而不是让它变成一个空的 401
    if not credential and not base_url:
        raise LLMConfigError("没有配置 API Key。到「设置 → 模型」里填写。")

    # 如果之前实测过能力，就直接用那份结果；否则用按端点推断的保守默认值。
    # 复用探测结果是为了不重复花钱——探测要发好几次真实请求。
    caps = _load_capabilities(settings)

    if provider_name == "openai_compatible":
        from .openai_provider import OpenAICompatProvider

        return OpenAICompatProvider(
            model=model,
            api_key=credential,
            base_url=base_url,
            max_tokens=int(settings.get("llm.max_tokens") or 16000),
            capabilities=caps,
        )

    from .anthropic_provider import AnthropicProvider

    # 凭据怎么发，取决于端点：官方用 x-api-key，第三方网关多数用
    # Authorization: Bearer。auto 模式按地址判断。
    auth_mode = settings.get("llm.auth_mode") or "auto"
    use_bearer = auth_mode == "bearer" or (
        auth_mode == "auto" and base_url and "api.anthropic.com" not in base_url.lower()
    )

    return AnthropicProvider(
        model=model,
        api_key="" if use_bearer else credential,
        auth_token=credential if use_bearer else "",
        base_url=base_url,
        max_tokens=int(settings.get("llm.max_tokens") or 16000),
        capabilities=caps,
    )


def _load_capabilities(settings):
    """读出之前探测并存下来的能力表。没有就返回 None（用默认值）。"""
    stored = settings.get("llm.capabilities")
    if not isinstance(stored, dict):
        return None

    # 端点或模型换了，旧的能力表就不可信了——能力是跟着端点走的
    if stored.get("base_url") != (settings.get("llm.base_url") or ""):
        log.debug("端点已变更，忽略旧的探测结果")
        return None
    if stored.get("model") != settings.get("llm.deep_model"):
        log.debug("模型已变更，忽略旧的探测结果")
        return None

    payload = stored.get("capabilities")
    if not isinstance(payload, dict):
        return None

    try:
        return Capabilities(
            **{k: v for k, v in payload.items() if k in Capabilities.__dataclass_fields__}
        )
    except TypeError:
        log.warning("存储的能力表格式不对，已忽略", exc_info=True)
        return None


def save_capabilities(settings, provider, capabilities) -> None:
    """把探测结果存下来，连同它的适用范围（端点 + 模型）。

    存的时候一并记下 base_url 与模型名：能力是跟着这两者走的，
    换了端点还沿用旧结论会导致行为错误（该降级的地方走了原生路径）。
    """
    from datetime import UTC, datetime

    settings.set(
        "llm.capabilities",
        {
            "base_url": settings.get("llm.base_url") or "",
            "model": provider.model,
            "probed_at": datetime.now(UTC).isoformat(),
            "capabilities": capabilities.to_dict()
            if hasattr(capabilities, "to_dict")
            else capabilities,
        },
        updated_by="probe",
    )


def chat_provider():
    """对话用的 Provider（深度模型）。"""
    return get_provider(fast=False)


def fast_provider():
    """高频轻任务用的 Provider（轻量模型）。"""
    return get_provider(fast=True)


def is_configured() -> bool:
    """能不能用。界面据此决定是否显示「需要先配置模型」的提示。"""
    try:
        get_provider()
        return True
    except (LLMConfigError, LLMError):
        return False
    except Exception:
        log.debug("检查模型配置时出错", exc_info=True)
        return False


def configuration_status() -> dict:
    """配置状态摘要，供设置页与健康检查展示。"""
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    provider_name = settings.get("llm.provider")
    base_url = (settings.get("llm.base_url") or "").strip()

    has_credential = settings.has_secret("llm.api_key")
    deep_model = settings.get("llm.deep_model")
    fast_model = settings.get("llm.fast_model")

    problems = []
    if not deep_model:
        problems.append("没有配置深度阅读模型")
    if not has_credential and not base_url:
        problems.append("没有配置 API Key")

    return {
        "provider": provider_name,
        "base_url": base_url or "（官方端点）",
        "deep_model": deep_model,
        "fast_model": fast_model,
        "has_credential": has_credential,
        "max_tokens": settings.get("llm.max_tokens"),
        "ready": not problems,
        "problems": problems,
    }


__all__ = [
    "EXTRACTION_TOOL",
    "Capabilities",
    "LLMConfigError",
    "LLMError",
    "LLMResponse",
    "Usage",
    "build_extraction_tool",
    "chat_provider",
    "configuration_status",
    "fast_provider",
    "get_provider",
    "is_configured",
    "save_capabilities",
]
