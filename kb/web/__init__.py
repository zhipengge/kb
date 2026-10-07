"""网页端（服务端渲染）。

**没有前端框架、没有构建步骤。** 模板由 Jinja 渲染，交互用原生 JS
（`static/js/` 下按关注点拆成若干小文件，经 base.html 的 `scripts` 块按页加载）。
需要画图时引 ECharts、看 PDF 时引 PDF.js、渲染公式引 KaTeX、编辑笔记引 Vditor。

> 早期文档里写的是「用少量的 Alpine.js / HTMX」。**实际从未用过**：
> Alpine 根本没下载，HTMX 虽然加载了但全站没有一个 `hx-` 属性。
> 现在这句话已经改掉——照着过时的文档去猜技术栈，比没有文档更浪费时间。

第三方库都以静态文件形式随仓库提供（见 `scripts/vendor_assets.py`），
不依赖 CDN —— 知识库常常部署在没有外网的内网机器上，一个引 CDN 的页面
在那里会白屏，而且是最难排查的那种「本地跑得好好的」问题。
"""

from __future__ import annotations

from flask import Blueprint

web_bp = Blueprint("web", __name__)

from . import chat, views  # noqa: E402,F401  — 导入即注册路由

__all__ = ["web_bp"]
