"""从 Claude Code 的配置里导入模型设置。

Claude Code 把服务商配置放在 ``~/.claude.json`` 的 ``env`` 段里，
形如::

    {
      "env": {
        "ANTHROPIC_AUTH_TOKEN": "sk-…",
        "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
        "ANTHROPIC_MODEL": "deepseek-flash[1m]",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-flash"
      }
    }

本机已经在用 Claude Code 的话，这些配置是现成的、有效的，让知识库复用它
比让用户再抄一遍密钥合理得多。

**但这是显式操作，不是自动的。** 导入会读取用户主目录下的密钥文件并写进
知识库的数据库，这种事必须由用户主动触发（设置页按钮或 CLI 命令），
不能在启动时悄悄做掉。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .base import LLMConfigError

log = logging.getLogger(__name__)

CLAUDE_CONFIG = "~/.claude.json"

# 从 Claude Code 配置到知识库设置的字段映射。
#
# 注意大小案例外：官方端点用 x-api-key，而 Claude Code 配置第三方网关时
# 用的是 AUTH_TOKEN（走 Authorization: Bearer）。所以导入时要把认证方式
# 一并推断出来，否则会拿着正确的 key 收到 401。
_ENV_KEYS = {
    "base_url": "ANTHROPIC_BASE_URL",
    "auth_token": "ANTHROPIC_AUTH_TOKEN",
    "api_key": "ANTHROPIC_API_KEY",
    "model": "ANTHROPIC_MODEL",
    "sonnet_model": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "opus_model": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "haiku_model": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
}


class ClaudeConfigError(LLMConfigError):
    """读取 Claude Code 配置失败。"""


def read_claude_config(path: str | os.PathLike[str] | None = None) -> dict:
    """读取并解析配置文件。找不到或格式不对时抛出可读的错误。"""
    config_path = Path(os.path.expanduser(str(path or CLAUDE_CONFIG)))

    if not config_path.is_file():
        raise ClaudeConfigError(
            f"找不到 {config_path}。\n"
            "这个文件由 Claude Code 维护，只有在用 Claude Code 且配置过自定义"
            "服务商时才存在。请在设置页手动填写模型配置。"
        )

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClaudeConfigError(f"读取 {config_path} 失败：{exc}") from exc

    if not isinstance(raw, dict):
        raise ClaudeConfigError(f"{config_path} 的内容不是一个对象")

    env = raw.get("env")
    if not isinstance(env, dict) or not env:
        raise ClaudeConfigError(
            f"{config_path} 里没有 env 段。\n"
            "这说明 Claude Code 用的是默认官方端点（或 OAuth 登录），"
            "没有可导入的自定义服务商配置。"
        )

    return env


def extract_settings(env: dict) -> dict:
    """把 Claude Code 的 env 段翻译成知识库的设置项。

    返回值直接可以交给 ``Settings.update_many``。
    """
    base_url = (env.get(_ENV_KEYS["base_url"]) or "").strip() or None
    auth_token = (env.get(_ENV_KEYS["auth_token"]) or "").strip()
    api_key = (env.get(_ENV_KEYS["api_key"]) or "").strip()
    credential = auth_token or api_key

    if not credential:
        raise ClaudeConfigError(
            f"配置里没有找到 API Key（{_ENV_KEYS['auth_token']} 或 "
            f"{_ENV_KEYS['api_key']} 都是空的）。"
        )

    # 主模型优先取 ANTHROPIC_MODEL；没设就退到 sonnet / opus 的默认值
    deep_model = (
        env.get(_ENV_KEYS["model"])
        or env.get(_ENV_KEYS["sonnet_model"])
        or env.get(_ENV_KEYS["opus_model"])
        or ""
    ).strip()

    fast_model = (env.get(_ENV_KEYS["haiku_model"]) or "").strip() or deep_model

    if not deep_model:
        raise ClaudeConfigError("配置里没有找到模型名（ANTHROPIC_MODEL 等字段都是空的）。")

    is_official = bool(base_url) and "api.anthropic.com" in base_url.lower()

    settings: dict = {
        # 所有 anthropic 兼容端点都走 anthropic provider——区别只在能力和认证方式
        "llm.provider": "anthropic",
        "llm.base_url": base_url or "",
        "llm.deep_model": deep_model,
        "llm.fast_model": fast_model,
        # 第三方网关用 Bearer，官方用 x-api-key
        "llm.auth_mode": "api_key" if (not base_url or is_official) else "bearer",
        # 思考会和正文抢输出预算，默认给宽一点
        "llm.max_tokens": 16000,
    }

    return settings, credential


def import_from_claude_config(
    settings,
    *,
    path: str | os.PathLike[str] | None = None,
    updated_by: str = "import",
) -> dict:
    """执行导入并落库。返回导入结果摘要（**不含密钥明文**）。"""
    env = read_claude_config(path)
    values, credential = extract_settings(env)

    for key, value in values.items():
        settings.set(key, value, updated_by=updated_by)
    # 密钥单独走加密写入
    settings.set("llm.api_key", credential, updated_by=updated_by)

    result = {
        "base_url": values["llm.base_url"] or "（官方端点）",
        "deep_model": values["llm.deep_model"],
        "fast_model": values["llm.fast_model"],
        "auth_mode": values["llm.auth_mode"],
        "credential": settings.mask(credential),
    }
    log.info(
        "已从 Claude Code 配置导入模型设置：%s / %s",
        result["base_url"], result["deep_model"],
    )
    return result


def preview(path: str | os.PathLike[str] | None = None) -> dict:
    """只读预览：导入会得到什么。密钥只显示掩码。"""
    env = read_claude_config(path)
    values, credential = extract_settings(env)
    return {
        "base_url": values["llm.base_url"] or "（官方端点）",
        "deep_model": values["llm.deep_model"],
        "fast_model": values["llm.fast_model"],
        "auth_mode": values["llm.auth_mode"],
        "credential": (credential[:7] + "…" + credential[-4:]) if len(credential) > 12 else "…",
    }


__all__ = [
    "CLAUDE_CONFIG",
    "ClaudeConfigError",
    "extract_settings",
    "import_from_claude_config",
    "preview",
    "read_claude_config",
]
