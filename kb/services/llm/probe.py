"""服务商能力探测。

为什么需要它：**Anthropic 兼容端点会「接受」它并不支持的参数**。

实测 DeepSeek 的兼容端点：
  * 传 ``cache_control`` → 不报错，但 ``cache_read_input_tokens`` 恒为 0；
  * 传 ``output_config.format`` → 不报错，但输出被整个吞掉，只剩思考；
  * 传 PDF 文档块 → 不报错，模型回答「无法确定」。

也就是说，**「没报错」完全不等于「生效了」**。如果只靠异常来判断，
这些特性会静默失效，而失效的表现是「AI 偶尔答非所问」这种极难排查的症状。

所以这里都设计成「有可观测的证据才算支持」：
  * 缓存 → 看 cache_read_input_tokens 是否 > 0；
  * 结构化输出 → 看返回的文本能不能解析成 JSON；
  * PDF → 看模型能否答出只有读了 PDF 才知道的内容。
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
from typing import Any

from .. import budget
from .base import EXTRACTION_TOOL, LLMError, build_extraction_tool

log = logging.getLogger(__name__)

# 探测用的最小 PDF：一份两页的假论文，内容是我们自己构造的，
# 所以「模型能不能答对」是明确的判据。
_PROBE_DOC_TITLE = "Sparse Routing with Learned Capacity"


def _probe_result(supported: bool, evidence: str) -> dict:
    return {"supported": supported, "evidence": evidence}


def _make_probe_png() -> str:
    """生成一张有明确内容的图，用于视觉探测。

    用 PIL 现场画而不是内置 base64 常量：手写的 PNG 十六进制极易出错，
    而「图片格式非法」和「不支持视觉」在错误信息上很难区分——
    我第一次探测就踩了这个坑，得到的 400 其实是我的图片有问题。
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 90, 90], fill="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode()


def _safe_call(provider, func, *args, **kwargs) -> tuple[Any, str | None]:
    """调用并吞掉异常，返回 ``(结果, 错误信息)``。

    探测过程里的失败是**预期的结果**，不是异常——「这个能力不支持」
    和「这次调用炸了」对调用方是同一件事：不要用它。
    """
    try:
        # 能力探测一次要发好几条真实请求，单独归一类——
        # 否则它会被混进「其它」，而用户完全看不出钱花到哪了。
        with budget.track("probe"):
            return func(*args, **kwargs), None
    except budget.BudgetExceeded:
        # **必须抢在 LLMError 之前重新抛出。**
        # 超预算会被下面那个 except 当成「这个能力不支持」记进探测报告，
        # 于是用户看到的结论是「你的网关不解析 PDF」——而真相是钱花完了。
        # 这正是「静默降级让结果变差看起来像模型不行」的典型形态。
        raise
    except LLMError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def probe_capabilities(provider, *, include_pdf: bool = True) -> dict:
    """实测服务商能力。

    会发起若干次真实调用，**消耗额度**。因此保持每次调用的输入输出都很小，
    并且把较贵的 PDF 探测做成可关闭的。
    """
    # dataclasses.replace 会做一层浅拷贝，但 notes 是 list——必须显式复制，
    # 否则下面的 append 会改到 provider 持有的那份能力表上
    caps = dataclasses.replace(provider.capabilities, notes=list(provider.capabilities.notes))
    results: dict[str, dict] = {}
    errors: list[str] = []

    # ---- 1. 基本可用性 ----
    response, error = _safe_call(
        provider, provider.complete,
        [{"role": "user", "content": "只回复两个字：收到"}],
        max_tokens=2000,
    )
    if error:
        errors.append(f"基本调用失败：{error}")
        return {
            "capabilities": caps.to_dict(),
            "checks": results,
            "errors": errors,
            "fatal": True,
        }

    results["basic"] = _probe_result(True, f"模型回显 {response.model}")
    caps.returns_thinking = bool(response.thinking)
    if caps.returns_thinking:
        results["thinking"] = _probe_result(
            True, f"默认返回思考（{len(response.thinking)} 字符）"
        )

    # ---- 2. 工具调用 ----
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    tools = [build_extraction_tool(schema, "提交城市名")]
    tool_response, tool_error = _safe_call(
        provider, provider.complete,
        [{"role": "user", "content": "用工具提交城市「北京」，不要用文字回答。"}],
        tools=tools, max_tokens=3000,
    )
    if tool_error or tool_response.first_tool_input(EXTRACTION_TOOL) is None:
        caps.tool_use = False
        results["tool_use"] = _probe_result(
            False, tool_error or f"未产生工具调用（stop={getattr(tool_response,'stop_reason',None)}）"
        )
    else:
        results["tool_use"] = _probe_result(
            True, f"工具参数 {json.dumps(tool_response.first_tool_input(EXTRACTION_TOOL), ensure_ascii=False)}"
        )

    # ---- 3. 视觉 ----
    png = _make_probe_png()
    vision_response, vision_error = _safe_call(
        provider, provider.complete,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": png},
                    },
                    {"type": "text", "text": "图中有一个红色方块吗？只答「是」或「否」。"},
                ],
            }
        ],
        max_tokens=3000,
    )
    if vision_error:
        caps.vision = False
        results["vision"] = _probe_result(False, vision_error)
    else:
        text = vision_response.text
        caps.vision = "是" in text or "yes" in text.lower()
        # 答「否」也说明能看图，只是识别错误；真正的失败是空回复或报错
        if not text:
            caps.vision = False
            results["vision"] = _probe_result(False, "返回了空文本，可能并未处理图片")
        else:
            results["vision"] = _probe_result(True, f"对图片的回答：{text[:40]!r}")

    # ---- 4. 提示缓存 ----
    # 判据是 cache_read_input_tokens > 0，而不是「参数被接受」
    long_system = "这是一段用于探测提示缓存的长文本。" * 120
    _, err1 = _safe_call(
        provider, provider.complete,
        [{"role": "user", "content": "回复 ok"}],
        system=long_system, max_tokens=1000,
    )
    second, err2 = _safe_call(
        provider, provider.complete,
        [{"role": "user", "content": "回复 ok"}],
        system=long_system, max_tokens=1000,
    )
    if err1 or err2:
        caps.prompt_cache = False
        results["prompt_cache"] = _probe_result(False, err1 or err2 or "")
    elif second.usage.cache_read_tokens > 0:
        caps.prompt_cache = True
        results["prompt_cache"] = _probe_result(
            True, f"第二次调用命中缓存 {second.usage.cache_read_tokens} tokens"
        )
    else:
        caps.prompt_cache = False
        results["prompt_cache"] = _probe_result(
            False, "请求被接受但缓存命中恒为 0（参数被忽略）"
        )

    # ---- 5. 结构化输出 ----
    try:
        structured = provider.complete(
            [{"role": "user", "content": '把 {"name":"kb","version":1} 原样返回为 JSON'}],
            max_tokens=2000,
            **({"output_config": {"format": {"type": "json_schema", "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "version": {"type": "integer"}},
                "required": ["name", "version"], "additionalProperties": False}}}}
               if provider.name == "anthropic" else {}),
        )
        try:
            json.loads(structured.text)
            caps.structured_output = True
            results["structured_output"] = _probe_result(True, "返回了可解析的 JSON")
        except (json.JSONDecodeError, TypeError):
            caps.structured_output = False
            results["structured_output"] = _probe_result(
                False, f"输出不可解析（{len(structured.text)} 字符文本）"
            )
    except Exception as exc:
        caps.structured_output = False
        results["structured_output"] = _probe_result(False, str(exc)[:120])

    # ---- 6. PDF 原生解析 ----
    if include_pdf:
        pdf_result = _probe_pdf(provider)
        caps.pdf_native = pdf_result["supported"]
        results["pdf_native"] = pdf_result

    caps.probed = True
    caps.notes = [note for note in caps.notes if "未启用" not in note]
    if not caps.prompt_cache:
        caps.notes.append("提示缓存不可用：多轮对话会重复计费，建议靠检索控制上下文长度。")
    if not caps.pdf_native:
        caps.notes.append("不支持 PDF 原生解析：论文正文由本地解析后以文本注入。")
    if not caps.structured_output:
        caps.notes.append("不支持原生结构化输出：抽取任务走工具调用。")

    return {"capabilities": caps.to_dict(), "checks": results, "errors": errors, "fatal": False}


def _probe_pdf(provider) -> dict:
    """探测 PDF 文档块是否真的被解析。

    判据是「模型能否答出只有读了 PDF 才知道的内容」。用一个我们自己生成的
    小 PDF，里面放一句独一无二的话。
    """
    try:
        import pymupdf
    except ImportError:
        return _probe_result(False, "本机没有 PyMuPDF，跳过")

    try:
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 100), _PROBE_DOC_TITLE, fontsize=16)
        page.insert_text((72, 140), "The secret marker is ZEBRA-7741.", fontsize=11)
        payload = document.tobytes()
        document.close()
    except Exception as exc:
        return _probe_result(False, f"构造探测用 PDF 失败：{exc}")

    encoded = base64.standard_b64encode(payload).decode()
    response, error = _safe_call(
        provider, provider.complete,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": encoded,
                        },
                    },
                    {"type": "text", "text": "这份 PDF 里的密文标记是什么？只回答标记本身。"},
                ],
            }
        ],
        max_tokens=3000,
    )
    if error:
        return _probe_result(False, error)

    # 只有答对那个独一无二的标记，才能确认它真的读了 PDF
    if "ZEBRA-7741" in response.text.upper():
        return _probe_result(True, "正确读出了 PDF 中的内容")
    return _probe_result(
        False, f"未能读出内容（回答：{response.text[:50]!r}），PDF 块被接受但未解析"
    )


__all__ = ["probe_capabilities"]
