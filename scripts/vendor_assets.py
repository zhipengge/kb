#!/usr/bin/env python
"""把前端依赖下载到 kb/web/static/vendor/。

**为什么不用 CDN。** 知识库常常部署在没有外网的机器上（内网服务器、
隔离环境），一个引 CDN 的页面在那里会直接白屏，而且是最难排查的那种
「本地跑得好好的」问题。把库随仓库带着，代价是几 MB 的静态文件。

用法::

    pipenv run python scripts/vendor_assets.py          # 下载缺失的
    pipenv run python scripts/vendor_assets.py --force  # 全部重新下载
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

VENDOR_DIR = Path(__file__).resolve().parent.parent / "kb" / "web" / "static" / "vendor"

# 每个库的选择理由：
#
#   cytoscape  —— 知识图谱。调研结论：它是「需要图算法与多种布局」时最合适的
#                 选择（自带最短路径、中心性、BFS/DFS 与多种布局算法）。
#                 ECharts 的 graph 模块偏通用图表，复杂关系图定制起来吃力。
#   echarts    —— 统计图表（仪表盘、时间线）。它的强项在这里，不在关系图。
#   katex      —— 公式渲染。论文公式以原始 LaTeX 保存，必须能渲染出来，
#                 否则 $...$ 就是一串符号。支持服务端预渲染，首屏无闪烁。
#   pdfjs      —— PDF 阅读器。支持 Range 请求，大文件可以边下边看。
#   htmx       —— 局部更新与 SSE。体积小，避免为此引入完整的 SPA 框架。
# 每个文件给多个镜像。实测部署环境里 cdn.jsdelivr.net 可能不可达
# （TLS 握手直接被切断），所以不能只依赖单一 CDN。
# unpkg 与 cdnjs 的目录结构不同，所以逐个列出完整 URL 而不是拼路径。
ASSETS: dict[str, tuple[list[str], str]] = {
    "cytoscape.min.js": ([
        "https://unpkg.com/cytoscape@3.33.2/dist/cytoscape.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.33.2/cytoscape.min.js",
    ], "知识图谱可视化"),
    "echarts.min.js": ([
        "https://unpkg.com/echarts@5.6.0/dist/echarts.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/echarts/5.6.0/echarts.min.js",
    ], "统计图表"),
    "katex.min.js": ([
        "https://unpkg.com/katex@0.16.11/dist/katex.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js",
    ], "公式渲染"),
    "katex.min.css": ([
        "https://unpkg.com/katex@0.16.11/dist/katex.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.css",
    ], "公式样式"),
    "katex-auto-render.min.js": ([
        "https://unpkg.com/katex@0.16.11/dist/contrib/auto-render.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/contrib/auto-render.min.js",
    ], "公式自动渲染"),
    "htmx.min.js": ([
        "https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/htmx/2.0.4/htmx.min.js",
    ], "局部更新与 SSE"),
    "pdf.min.mjs": ([
        "https://unpkg.com/pdfjs-dist@4.10.38/build/pdf.min.mjs",
    ], "PDF 阅读器"),
    "pdf.worker.min.mjs": ([
        "https://unpkg.com/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs",
    ], "PDF 解析 worker"),
}

# KaTeX 的字体文件。CSS 里引用了它们，缺了会导致公式用系统字体凑合显示，
# 分数、根号、上下标全都变形——比不渲染还难看出问题。
KATEX_FONTS = [
    "KaTeX_Main-Regular.woff2", "KaTeX_Main-Bold.woff2", "KaTeX_Main-Italic.woff2",
    "KaTeX_Math-Italic.woff2", "KaTeX_Math-BoldItalic.woff2",
    "KaTeX_Size1-Regular.woff2", "KaTeX_Size2-Regular.woff2",
    "KaTeX_Size3-Regular.woff2", "KaTeX_Size4-Regular.woff2",
    "KaTeX_AMS-Regular.woff2", "KaTeX_Caligraphic-Regular.woff2",
    "KaTeX_Fraktur-Regular.woff2", "KaTeX_SansSerif-Regular.woff2",
    "KaTeX_Script-Regular.woff2", "KaTeX_Typewriter-Regular.woff2",
]


# ---------------------------------------------------------------------------
# Vditor（笔记编辑器）
#
# 它的 dist 有 21 MB，但绝大部分是**可选的渲染器**：MathJax 6.5MB、mermaid 3MB、
# graphviz 1.9MB、echarts 1MB、abcjs/smiles/flowchart/plantuml/markmap……
# 我们只用 KaTeX，所以按清单挑选，实际只带约 6.5 MB。
#
# 唯一省不掉的是 lute（3.7MB）——它是 Markdown 引擎本体，体积大是因为
# 编译自 Go，且只在编辑器真正启用时才加载，不影响其它页面速度。
#
# **必须把 Vditor 的 ``cdn`` 选项指到本地目录**：它默认从 unpkg 懒加载
# lute、图标、KaTeX，内网部署下会表现为「编辑器出来了但一输入就卡住」，
# 而且不报错，只是资源永远加载不完。这是本文件存在的意义本身。
VDITOR_VERSION = "3.10.9"
VDITOR_BASE = f"https://unpkg.com/vditor@{VDITOR_VERSION}/dist"

# 精确文件
VDITOR_FILES = (
    "index.min.js",
    "index.css",
    # 提供 Vditor.preview()：把 Markdown 渲染成 HTML（含公式、代码高亮）。
    # 会话页要在浏览器里渲染流式到达的 Markdown，用它才能和编辑器
    # 用同一套渲染规则——自己再引一个 marked.js 迟早两边渲染结果不一致。
    "method.min.js",
    "js/lute/lute.min.js",
    "js/icons/ant.js",              # 只带一套图标（另一套 material 用不上）
    "js/i18n/zh_CN.js",
    "js/i18n/en_US.js",
    # Mermaid：在笔记里画算法流程图与模型架构图。
    # 之前跳过了它（3MB），结果是生成的笔记里只能有纯文字和公式，
    # 「这个方法的流程是什么」这种问题没法用图回答——而架构图恰恰是
    # 论文笔记里最该有、也最难用文字替代的东西。
    "js/mermaid/mermaid.min.js",
    "js/highlight.js/highlight.min.js",
    "js/highlight.js/styles/atom-one-light.min.css",
    "js/highlight.js/styles/atom-one-dark.min.css",
)
# 整目录（含 KaTeX 的 55 个字体文件，逐个硬编码不现实且升级必漏）
VDITOR_PREFIXES = (
    "js/katex/",
    "css/content-theme/",
    "images/emoji/",
)


def _vditor_file_list() -> list[str]:
    """取 Vditor 的文件清单，再按上面的规则挑选。

    向 unpkg 要清单而不是硬编码：KaTeX 目录里有 60 多个文件，
    硬编码的结果是升级版本时悄悄漏文件——而且报错发生在浏览器控制台，
    不是在这里。
    """
    url = f"https://unpkg.com/vditor@{VDITOR_VERSION}/dist/?meta"
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise SystemExit(
            f"取 Vditor 文件清单失败（{type(exc).__name__}）：{url}\n"
            "这一步需要联网。离线环境请手动把 Vditor 的 dist 放进 "
            f"{VENDOR_DIR / 'vditor'}。"
        ) from exc

    selected: list[str] = []
    for item in payload.get("files", []):
        if item.get("type") == "directory":
            continue
        path = item["path"].replace("/dist/", "", 1)
        if path in VDITOR_FILES or path.startswith(VDITOR_PREFIXES):
            selected.append(path)
    return sorted(selected)


def _fetch_one(url: str, target: Path) -> tuple[bool, str]:
    try:
        with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as response:
            response.raise_for_status()
            with open(target, "wb") as handle:
                for chunk in response.iter_bytes(64 * 1024):
                    handle.write(chunk)
    except Exception as exc:
        target.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}"

    size = target.stat().st_size
    if size < 512:
        # 明显不是正常文件——多半是错误页或重定向到了别处
        target.unlink(missing_ok=True)
        return False, f"内容异常（仅 {size} 字节）"
    return True, f"{size / 1024:.0f} KB"


def download(urls: list[str], target: Path, *, force: bool = False) -> tuple[bool, str]:
    """依次尝试多个镜像，第一个成功的即用。"""
    if target.exists() and not force:
        return True, f"已存在（{target.stat().st_size / 1024:.0f} KB）"

    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        ok, message = _fetch_one(url, target)
        if ok:
            host = url.split("/")[2]
            return True, message if len(urls) == 1 else f"{message}（来自 {host}）"
        errors.append(f"{url.split('/')[2]}: {message}")

    return False, "；".join(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载前端依赖到 static/vendor/")
    parser.add_argument("--force", action="store_true", help="已存在的也重新下载")
    args = parser.parse_args()

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    print(f"目标目录：{VENDOR_DIR}\n")
    for filename, (urls, purpose) in ASSETS.items():
        ok, message = download(urls, VENDOR_DIR / filename, force=args.force)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {filename:28} {purpose:16} {message}")
        if not ok:
            failures.append(filename)

    fonts_dir = VENDOR_DIR / "fonts"
    for font in KATEX_FONTS:
        urls = [
            f"https://unpkg.com/katex@0.16.11/dist/fonts/{font}",
            f"https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/fonts/{font}",
        ]
        ok, _ = download(urls, fonts_dir / font, force=args.force)
        if not ok:
            failures.append(f"fonts/{font}")
    print(f"  {'✓' if not failures else '✗'} KaTeX 字体 {len(KATEX_FONTS)} 个"
          f"{'（有缺失）' if failures else ''}")

    # ---- Vditor（文件多，只打汇总，失败才逐条列） ----
    vditor_dir = VENDOR_DIR / "vditor"
    try:
        vditor_files = _vditor_file_list()
    except SystemExit as exc:
        print(f"  ✗ Vditor：{exc}")
        failures.append("vditor（清单获取失败）")
        vditor_files = []

    # 落盘布局要保持 npm 包原样（dist/ 这一层不能少）：
    # Vditor 运行时拼的是 `options.cdn + "/dist/js/..."`，少一层 dist
    # 就会表现为「编辑器出来了，但 lute 永远加载不完」——不报错，只是卡住。
    vditor_failed: list[str] = []
    for path in vditor_files:
        ok, _ = download([f"{VDITOR_BASE}/{path}"], vditor_dir / "dist" / path, force=args.force)
        if not ok:
            vditor_failed.append(path)
            failures.append(f"vditor/dist/{path}")

    if vditor_files:
        added = sum(
            f.stat().st_size for f in vditor_dir.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        )
        mark = "✓" if not vditor_failed else "✗"
        print(f"  {mark} {'vditor/':28} {'笔记编辑器':16} "
              f"{len(vditor_files) - len(vditor_failed)}/{len(vditor_files)} 个文件 · "
              f"{added / 1024 / 1024:.1f} MB")

    if failures:
        print(f"\n有 {len(failures)} 个文件未下载成功：")
        for name in failures:
            print(f"  · {name}")
        print("\n可以重跑一次；公式与图谱缺资源时会降级显示，不影响其它功能。")
        sys.exit(1)

    total = sum(f.stat().st_size for f in VENDOR_DIR.rglob("*") if f.is_file())
    print(f"\n全部就绪，共 {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
