"""LaTeX 源码的获取与解析。

**为什么优先用源码而不是 PDF。** 论文的 PDF 是**排版结果**，结构信息在排版时
就被压掉了；要从它反推出「这是第几节、这是公式、这是图注」只能靠字号、缩进、
位置这些启发式。而 LaTeX 源码里这些信息是**显式声明**的：

    \\section{Method}          -> 章节层级，确定无疑
    \\begin{equation}...\\end{equation}  -> 公式边界，确定无疑
    \\caption{...}             -> 图注，确定无疑
    \\cite{vaswani2017}        -> 引用键，确定无疑

最后一条尤其重要：**引用键是从 PDF 里拿不到的**——PDF 里只有 "[1]"，而
"[1]" 到具体论文的映射要靠解析参考文献列表再加猜测。源码里直接就有键名。

arXiv 为绝大多数论文提供源码包（``https://arxiv.org/e-print/<id>``）。
拿不到时才退回 PDF 解析。
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import re
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .pdf import normalize_arxiv

log = logging.getLogger(__name__)

# arXiv 的抓取礼仪：官方要求请求间隔约 3 秒，并带可识别的 User-Agent。
# 遵守它不只是礼貌——被限流后整个功能会不可用。
ARXIV_EPRINT = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_MIN_INTERVAL = 3.0
USER_AGENT = "kb-knowledge-base/0.1 (personal research paper manager)"

_last_request_at = 0.0

# 主文档里通常会在 preamble 声明这些，用来判断哪个 .tex 是入口
_MAIN_DOC_MARKERS = (r"\documentclass", r"\begin{document}")


class LatexError(RuntimeError):
    """LaTeX 源码获取或解析失败。"""


@dataclass
class LatexFigure:
    """源码里的一处图/表。"""

    kind: str  # figure / table
    caption: str = ""
    label: str | None = None
    graphics: list[str] = field(default_factory=list)  # \includegraphics 引用的文件
    env: str = ""
    # 表格正文（tabular 渲染成文本）。图表在源码里除了标题还有实体内容，
    # 而**数值全在正文里**——只留标题等于把整张表丢了。
    body: str = ""


@dataclass
class LatexEquation:
    latex: str
    label: str | None = None
    numbered: bool = True


@dataclass
class LatexSection:
    """源码里的一个章节。"""

    title: str
    level: int
    path: str
    label: str | None = None  # \label{sec:xxx}，用于解析 \ref 指向
    # 本节正文的原始 LaTeX（含公式与图表环境）。
    # 结构遍历时先切出这段原文，正文转换推迟到真正需要时再做——
    # 遍历阶段就要转的话，任何一处转换异常都会毁掉整篇的结构解析。
    raw_body: str = ""
    paragraphs: list[str] = field(default_factory=list)
    equations: list[LatexEquation] = field(default_factory=list)
    figures: list[LatexFigure] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


@dataclass
class LatexDocument:
    """解析后的源码文档。"""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    sections: list[LatexSection] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    bibitems: dict[str, str] = field(default_factory=dict)  # key -> 原始条目
    source_dir: Path | None = None
    main_file: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.sections) or bool(self.abstract)


# --------------------------------------------------------------------------
# 获取
# --------------------------------------------------------------------------


def _polite_get(
    url: str,
    *,
    timeout: float = 60.0,
    total_timeout: float = 240.0,
    max_bytes: int = 64 * 1024 * 1024,
):
    """带速率限制的抓取。

    用流式读取并限制总大小：arXiv 的源码包偶尔会很大（含大量图片），
    一次性读进内存可能把进程撑爆。

    **必须同时有 per-read 超时和总时限。** httpx 的 ``timeout`` 管的是
    「两次读到数据之间的最大间隔」，不是整个请求的时长。实测 arXiv 触发
    限流时会以极慢的速度滴数据（每几十秒一两个字节），per-read 超时永远
    不触发，下载可以无限期挂着。

    后果不是报错而是**静默卡死**：批量任务跑到某一篇就停住，进程还活着、
    CPU 空闲、日志没有任何异常——只能靠人去发现「怎么不动了」。
    """
    global _last_request_at

    import httpx

    elapsed = time.monotonic() - _last_request_at
    if elapsed < ARXIV_MIN_INTERVAL:
        time.sleep(ARXIV_MIN_INTERVAL - elapsed)

    try:
        with httpx.stream(
            "GET", url, timeout=timeout, follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as response:
            _last_request_at = time.monotonic()
            if response.status_code == 404:
                return None, "论文没有提供源码"
            if response.status_code == 403:
                return None, "被 arXiv 拒绝访问（可能触发限流，稍后再试）"
            response.raise_for_status()

            deadline = time.monotonic() + total_timeout
            buffer = io.BytesIO()
            for chunk in response.iter_bytes(64 * 1024):
                if time.monotonic() > deadline:
                    raise LatexError(
                        f"下载超过 {total_timeout:.0f} 秒仍未完成（已收到 "
                        f"{buffer.tell() // 1024} KB），判定为被限流，已放弃"
                    )
                buffer.write(chunk)
                if buffer.tell() > max_bytes:
                    raise LatexError(f"源码包超过 {max_bytes // 1024 // 1024} MB，已放弃")
            return buffer.getvalue(), None
    except LatexError:
        raise
    except Exception as exc:
        return None, f"下载失败：{exc}"


def fetch_arxiv_source(arxiv_id: str, dest_dir: Path) -> tuple[Path | None, str]:
    """下载并解开 arXiv 源码包。

    返回 ``(源码目录, 错误信息)``。源码目录里会有若干 .tex 与图片文件。

    注意 arXiv 的 ``e-print`` 端点返回的内容**有三种可能**，必须都处理：
      * gzip 压缩的 tar 包（最常见，多文件论文）
      * gzip 压缩的单个 .tex（单文件投稿）
      * 一个裸 PDF（作者只传了 PDF，没有源码）
    """
    normalized = normalize_arxiv(arxiv_id)
    if not normalized:
        return None, "没有 arXiv 编号"

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = ARXIV_EPRINT.format(arxiv_id=normalized)
    payload, error = _polite_get(url)
    if payload is None:
        return None, error or "未知错误"

    if not payload:
        return None, "返回内容为空"

    # 裸 PDF：作者没提供源码
    if payload[:5] == b"%PDF-":
        return None, "作者没有提供源码，只上传了 PDF"

    data = payload
    # gzip 解压（tar 包和单个文件都是 gzip 的）
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except OSError as exc:
            return None, f"gzip 解压失败：{exc}"

    # tar 包
    if tarfile.is_tarfile(io.BytesIO(data)):
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                _safe_extract(archive, dest_dir)
        except (tarfile.TarError, OSError) as exc:
            return None, f"解包失败：{exc}"
    else:
        # 单个 .tex 文件
        (dest_dir / "main.tex").write_bytes(data)

    return dest_dir, ""


def _safe_extract(archive: tarfile.TarFile, dest: Path) -> None:
    """解包，拒绝会逃出目标目录的成员。

    源码包是**外部输入**（来自 arXiv 的第三方上传）。tar 成员名里的 ``..``
    或绝对路径可以让文件被写到任意位置——这是 tar 解包的经典漏洞。
    Python 3.12+ 的 ``filter='data'`` 就是干这个的，但为了兼容旧版本，
    这里自己做一遍检查。
    """
    dest_resolved = dest.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            # 符号链接同样能指向目录外，直接跳过
            continue
        target = (dest_resolved / member.name).resolve()
        if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
            log.warning("跳过越界的归档成员：%s", member.name)
            continue
        if member.isfile():
            member.name = os.path.relpath(target, dest_resolved)
            archive.extract(member, dest_resolved, set_attrs=False)


def find_main_tex(source_dir: Path) -> Path | None:
    """找出主 .tex 文件。

    判据优先 ``\\begin{document}``（正文起始，只有入口文件才有），
    其次是 ``\\documentclass``，再次是文件名 ``main.tex``/``paper.tex``，
    最后退到最大的那个 .tex——投稿包里的附件（response letter、
    补充材料）通常比正文小。
    """
    # 只检查**相对于源码目录**的路径组成部分。
    # 用绝对路径的 parts 会连祖先目录一起检查——而缓存目录通常在
    # ~/.local/share/... 下面，".local" 以点开头，那样会把所有文件都过滤掉。
    # 本意只是跳过源码包内部的隐藏目录（.git、.cache 之类）。
    candidates = []
    for path in source_dir.rglob("*.tex"):
        try:
            relative = path.relative_to(source_dir)
        except ValueError:
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        candidates.append(path)

    if not candidates:
        return None

    scored: list[tuple[int, int, Path]] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = 0
        if r"\begin{document}" in text:
            score += 100
        if r"\documentclass" in text:
            score += 50
        if path.name.lower() in {"main.tex", "paper.tex", "article.tex", "ms.tex"}:
            score += 20
        # 章节命令的数量也是「这是正文」的强信号
        score += min(30, len(re.findall(r"\\(?:sub)*section\*?\{", text)))
        scored.append((score, len(text), path))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = scored[0]
    log.debug("主文档判定：%s（评分 %d）", best[2].name, best[0])
    return best[2]


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------


def _env_text(node) -> str:
    """取出一个 TexSoup 节点的**原始**文本内容。

    不用 ``node.string``：那个属性只对「纯文本子节点」有效，
    而公式环境里通常还有 ``\\label{}`` 等命令，会直接抛 AssertionError。
    """
    parts = []
    for child in node.contents:
        parts.append(str(child))
    return "".join(parts).strip()


def _clean_latex(text: str) -> str:
    """把 LaTeX 片段转成可读文本，但**保留数学**。

    两个关键取舍：

    1. **数学保留原样**（``$...$``），不转成 Unicode。因为前端用 KaTeX 渲染，
       ``\\frac{1}{2}`` 渲染出来是真正的分数，而转成 ``1/2`` 就再也变不回去了。
    2. **引用保留键名**，不转成 ``<cit.>``。``pylatexenc`` 默认会把
       ``\\cite{vaswani2017}`` 变成 ``<cit.>``，键名一丢，引用关系图就建不起来了。
       所以先把 ``\\cite{...}`` 抽出来替换成占位符，转换完再放回去。
    """
    import logging as _logging

    from pylatexenc.latex2text import LatexNodes2Text

    # pylatexenc 遇到无法解析的宏时会把提示直接打到 stderr（例如
    # "Open LaTeX blocks:"）。论文里自定义宏很常见，这些提示会刷屏，
    # 而它们对使用者没有意义——真正需要知道失败的地方我们会自己记日志。
    _logging.getLogger("pylatexenc").setLevel(_logging.ERROR)

    if not text:
        return ""

    # 1) 保护数学片段
    math_spans: list[str] = []

    def stash_math(match: re.Match) -> str:
        math_spans.append(match.group(0))
        return f"\x00MATH{len(math_spans) - 1}\x00"

    protected = re.sub(
        r"(\$\$?.+?\$\$?|\\\[.+?\\\]|\\\(.+?\\\))",
        stash_math,
        text,
        flags=re.S,
    )

    # 2) 保护引用键
    citations: list[str] = []

    def stash_cite(match: re.Match) -> str:
        keys = match.group(1)
        index = len(citations)
        citations.append(keys)
        return f"\x00CITE{index}\x00"

    protected = re.sub(r"\\cite[a-z]*\{([^}]*)\}", stash_cite, protected)
    protected = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", protected)

    # 3) 正文转 Unicode
    try:
        converter = LatexNodes2Text(math_mode="verbatim", keep_comments=False)
        text = converter.latex_to_text(protected)
    except Exception as exc:
        log.debug("LaTeX 文本转换失败，退回原文：%s", exc)
        text = protected

    # 4) 还原数学与引用
    for index, original in enumerate(math_spans):
        text = text.replace(f"\x00MATH{index}\x00", original)
    for index, keys in enumerate(citations):
        text = text.replace(f"\x00CITE{index}\x00", f"[{keys}]")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def resolve_includes(
    tex_text: str,
    source_dir: Path,
    *,
    depth: int = 0,
    seen: set[str] | None = None,
) -> tuple[str, list[str]]:
    """把 ``\\input{}`` / ``\\include{}`` 引用的文件内容展开进来。

    **这一步不能省。** 把正文拆成多个 .tex 再 ``\\input`` 进来是极为常见的
    投稿习惯（实测 Attention Is All You Need 就是：正文在 introduction.tex、
    model_architecture.tex 等文件里，ms.tex 只是骨架）。不展开的话，
    只能看到骨架里那几行 ``\\input``，子章节、公式、引用键全都会丢。

    返回 ``(展开后的文本, 警告列表)``。
    """
    if seen is None:
        seen = set()
    warnings: list[str] = []

    # 防止 A input B、B input A 造成无限递归
    if depth > 16:
        return tex_text, ["\\input 嵌套层数过深，已停止展开"]

    pattern = re.compile(r"\\(?:input|include)\{([^}]+)\}")

    def replace(match: re.Match) -> str:
        raw_name = match.group(1).strip()
        # \input{} 通常不带扩展名
        candidates = [raw_name]
        if not raw_name.endswith(".tex"):
            candidates.append(raw_name + ".tex")

        target: Path | None = None
        for candidate in candidates:
            path = (source_dir / candidate).resolve()
            # 防越界：源码包是外部输入，路径可能被构造成指到目录外
            try:
                path.relative_to(source_dir.resolve())
            except ValueError:
                warnings.append(f"跳过越界的 \\input：{candidate}")
                continue
            if path.is_file():
                target = path
                break

        if target is None:
            warnings.append(f"找不到 \\input 的文件：{raw_name}")
            return ""  # 用空串替换，保持文档结构完整

        key = str(target)
        if key in seen:
            return ""  # 已经展开过，避免重复内容
        seen.add(key)

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warnings.append(f"读取 {raw_name} 失败：{exc}")
            return ""

        expanded, nested = resolve_includes(
            content, target.parent, depth=depth + 1, seen=seen
        )
        warnings.extend(nested)
        return f"\n% ==== 来自 {target.name} ====\n{expanded}\n"

    # 简单循环替换即可：每次替换后文本变长，但 pattern 不会匹配到已展开的
    # 内容里（除非被展开的文件本身又含 \input，那由递归处理）
    previous = None
    result = tex_text
    for _ in range(20):
        if result == previous:
            break
        previous = result
        result = pattern.sub(replace, result)

    return result, warnings


def _strip_author_noise(text: str) -> str:
    """清掉作者字段里的脚注与注释。

    ``\\author{}`` 里常塞着 ``\\thanks{}``（"Equal contribution"、"Work done
    while at…"）以及被注释掉的作者名单。不清掉的话，作者列表会混进大段散文——
    实测 Attention 那篇就抽出了 "Ashish VaswaniEqual contribution. Listing
    order is r, om. Jakob proposed replacing RNNs…" 这种结果。
    """
    # \thanks{...} 支持一层嵌套花括号
    cleaned = re.sub(r"\\thanks\{((?:[^{}]|\{[^{}]*\})*)\}", "", text)
    # \footnote、\affiliation 之类同理
    cleaned = re.sub(r"\\(?:footnote|affiliation|affaddr|email|orcid)\{((?:[^{}]|\{[^{}]*\})*)\}", "", cleaned)
    # LaTeX 注释
    cleaned = re.sub(r"(?<!\\)%.*?$", "", cleaned, flags=re.M)
    # 作者之间常用 \and 分隔
    cleaned = re.sub(r"\\and\b", ", ", cleaned)
    return cleaned


# 这些环境里的 % 是内容的一部分（代码、URL、逐字文本），不能当注释剥掉
_VERBATIM_ENVS = ("verbatim", "verbatim*", "lstlisting", "minted", "Verbatim", "alltt")


def strip_comments(tex_text: str) -> tuple[str, int]:
    """剥掉 LaTeX 注释，返回 ``(清理后的文本, 剥掉的注释数)``。

    **这一步是正确性要求，不是美化。** 论文源码里通常留着大量被注释掉的
    草稿——实测 Attention Is All You Need 里就有一整段注释掉的公式
    （``%    A(\\kq, \\km, \\vm) = ...``）。不剥掉的话，这些**论文里并不存在**
    的内容会被建进索引，检索时会返回「论文没说过的话」。

    规则：``%`` 到行尾是注释，但 ``\\%`` 是转义的百分号，
    且 verbatim 类环境内的 ``%`` 是普通字符。
    """
    if "%" not in tex_text:
        return tex_text, 0

    lines = tex_text.split("\n")
    output: list[str] = []
    removed = 0
    verbatim_depth = 0

    for line in lines:
        # 跟踪 verbatim 类环境的进出
        stripped = line.strip()
        for env in _VERBATIM_ENVS:
            if stripped.startswith((f"\\begin{{{env}}}", f"\\begin{{{env}*}}")):
                verbatim_depth += 1
            elif stripped.startswith((f"\\end{{{env}}}", f"\\end{{{env}*}}")):
                verbatim_depth = max(0, verbatim_depth - 1)

        if verbatim_depth > 0:
            output.append(line)
            continue

        # 找到第一个未被转义的 %
        index = 0
        cut = -1
        while True:
            index = line.find("%", index)
            if index < 0:
                break
            # 数一下前面有几个连续的反斜杠，奇数个说明这个 % 被转义了
            backslashes = 0
            probe = index - 1
            while probe >= 0 and line[probe] == "\\":
                backslashes += 1
                probe -= 1
            if backslashes % 2 == 0:
                cut = index
                break
            index += 1

        if cut >= 0:
            output.append(line[:cut])
            if line[cut:].strip() != "%":
                removed += 1
        else:
            output.append(line)

    return "\n".join(output), removed


def resolve_refs(tex_text: str, sections: list[LatexSection]) -> str:
    """把 ``\\ref{sec:xxx}`` 换成可读的章节引用。

    论文里满篇 ``\\ref{sec:method}`` 这类交叉引用。不处理的话，正文里会留下
    ``[sec:method]`` 这样的标签——对读笔记的人来说是噪音，对模型来说则是
    一个无意义的 token 序列，还可能被误读成引文编号。

    解析不出来的标签**保持原样**而不是删掉：一个可见的 ``[eq:unknown]``
    至少说明「这里原本有个引用」，直接删掉会让句子读起来像原文就没有引用。
    """
    if "\\ref{" not in tex_text:
        return tex_text

    # label -> 章节标题。优先用标题，没有标题就退回标签本身
    mapping: dict[str, str] = {}
    for section in sections:
        if section.label:
            mapping[section.label] = section.title

    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        title = mapping.get(key)
        if title:
            return f"「{title}」"
        # 去掉常见前缀让标签稍微可读一点（sec:method -> method）
        readable = re.sub(r"^(?:sec|eq|tab|fig|app|alg):", "", key)
        return f"[{readable or key}]"

    return re.sub(r"\\ref\{([^}]*)\}", replace, tex_text)


# 宏定义的头部。**不用正则匹配宏体**——宏体里的花括号嵌套层数不定，
# 实测 `\newcommand{\algname}{{{UniAD}}}` 有三层，写死层数的正则直接漏掉，
# 而且漏得无声无息（宏没展开，方法名就没了）。改成扫描器按平衡括号取。
_MACRO_HEAD = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z@]+)\}?"
    r"(?:\s*\[(\d+)\])?(?:\s*\[[^\]]*\])?\s*\{"
)

# 只起格式作用、可以安全剥掉外层取内容的命令
_FORMATTING_WRAPPERS = re.compile(
    r"\\(?:textbf|textit|textrm|textsf|textsc|texttt|emph|mbox|hbox|"
    r"mathrm|mathbf|mathit|mathsf|text|ensuremath)\{((?:[^{}]|\{[^{}]*\})*)\}"
)
_COLOR_WRAPPER = re.compile(r"\\textcolor\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}")


def _read_group(text: str, start: int) -> tuple[str, int] | None:
    """从 ``text[start] == '{'`` 开始读一个**平衡**花括号组，返回 (内容, 结束位置)。"""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2  # 跳过转义（\{ \} 不算括号）
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
        index += 1
    return None


def _macro_text(body: str) -> str | None:
    """从宏体里取出可用的纯文本；取不出就返回 None（不展开）。"""
    if "#" in body:
        return None  # 带参数的宏，展开错的风险大于收益
    text = body
    for _ in range(6):
        unwrapped = _FORMATTING_WRAPPERS.sub(r"\1", text)
        unwrapped = _COLOR_WRAPPER.sub(r"\1", unwrapped)
        if unwrapped == text:
            break
        text = unwrapped
    text = re.sub(r"\\[A-Za-z]+", "", text)  # 剩下的命令直接丢掉
    text = text.replace("{", "").replace("}", "").strip()
    # 必须含真正的单词。否则 `\newcommand{\cmark}{\ding{51}}` 会被展开成 "51"，
    # 把表格里的勾选标记变成一堆散落的数字——那是往正文里灌噪声，不是救术语。
    if not re.search(r"[A-Za-z]{2,}", text):
        return None
    # 名字通常很短。太长的多半是整句文本，展开进正文会造成大量重复内容。
    return text if len(text) <= 60 else None


def expand_macros(tex_text: str) -> tuple[str, int]:
    """把自定义宏展开成它的文本内容，返回 (文本, 展开的宏个数)。

    **为什么必须做这一步。** 论文普遍把方法名定义成宏：

        \\newcommand{\\algname}{{{UniAD}}}

    实测 UniAD 那篇正文里 ``\\algname`` 出现 **61 次**，而字面的 "UniAD"
    只有 20 次。剥 LaTeX 命令时 ``\\algname`` 被整个删掉，于是
    **方法名在索引里凭空消失**——101 个分块里只剩 2 块含 "uniad"，
    而引用它的其它论文最多有 17 块。结果是搜 "UniAD" 找不到 UniAD 的论文；
    凡是把方法名写成宏的论文（比例相当高）都会是同样下场。

    方法名是论文最重要的检索词，这一步不能省。

    安全边界：只展开**无参数**的宏（``#1`` 这类直接跳过），宏体必须能
    取出纯文本且不超过 60 字符。宁可少展开几个，也不要把一整句正文
    展开到几十处，那会造成内容重复、把检索结果灌水。
    """
    definitions: dict[str, str] = {}
    position = 0
    while True:
        match = _MACRO_HEAD.search(tex_text, position)
        if match is None:
            break
        group = _read_group(tex_text, match.end() - 1)
        if group is None:
            position = match.end()
            continue
        body, position = group
        if match.group(2):  # 带参数的宏，跳过
            continue
        text = _macro_text(body)
        if text:
            definitions[match.group(1)] = text

    if not definitions:
        return tex_text, 0

    # 长名字放前面，避免 \alg 抢先匹配掉 \algname
    names = sorted(definitions, key=len, reverse=True)
    pattern = re.compile(
        r"\\(?P<name>" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z])\{?\}?"
    )
    replaced = pattern.sub(lambda m: definitions[m.group("name")], tex_text)
    return replaced, len(definitions)


def parse_latex(tex_text: str, *, source_dir: Path | None = None) -> LatexDocument:
    """解析一份 LaTeX 主文档。"""
    from TexSoup import TexSoup

    document = LatexDocument(source_dir=source_dir)

    # 先展开 \input/\include。必须在解析之前做——否则看到只是骨架，
    # 章节、公式、引用键全部拿不到。
    if source_dir is not None:
        tex_text, include_warnings = resolve_includes(tex_text, source_dir)
        document.errors.extend(include_warnings)

    # 再剥注释。必须早于所有内容提取——否则注释掉的草稿会被当成正文索引进去。
    tex_text, removed_comments = strip_comments(tex_text)
    if removed_comments:
        log.debug("剥离了 %d 处 LaTeX 注释", removed_comments)

    # 展开自定义宏。**必须早于所有命令剥离**，否则方法名会随宏一起被删掉。
    tex_text, expanded = expand_macros(tex_text)
    if expanded:
        log.debug("展开了 %d 个自定义宏", expanded)

    # TexSoup 对不完整/含自定义宏的文档常常抛异常。解析失败时用正则兜底——
    # 论文里自定义宏极其常见（\newcommand 出来的简写），不能因此整篇放弃。
    try:
        soup = TexSoup(tex_text, tolerance=1)
    except Exception as exc:
        document.errors.append(f"TexSoup 解析失败（{exc}），改用正则提取章节")
        document.sections = _sections_by_regex(tex_text)
        _extract_preamble_regex(tex_text, document)
        return document

    _extract_preamble(soup, document, tex_text)
    _extract_bibliography(soup, tex_text, document)

    # 只处理 \begin{document} 之后的正文。
    #
    # 用文本查找定位而不是 str(soup.find("document"))：后者是**重新序列化**，
    # 输出的文本与原文可能逐字不同（空白、注释、命令参数的写法），
    # 而下游的章节切分依赖精确偏移。差一个字符，所有章节的正文就会错位。
    body_text = _document_body(tex_text)

    document.sections = _walk_sections(body_text)
    if not document.sections:
        document.sections = _sections_by_regex(body_text)

    # 解析交叉引用。放在章节切分之后——需要先知道各章节的标签才能建立映射。
    if document.sections:
        resolved = resolve_refs(body_text, document.sections)
        if resolved != body_text:
            # 正文是按偏移从 body_text 里切出来的，替换后偏移会变，
            # 所以必须重新走一遍切分，否则章节内容又会错位
            rerun = _walk_sections(resolved)
            if rerun and len(rerun) == len(document.sections):
                for original, updated in zip(document.sections, rerun, strict=False):
                    original.raw_body = updated.raw_body

    # 收集全文引用键
    seen: list[str] = []
    for section in document.sections:
        for key in section.citations:
            if key not in seen:
                seen.append(key)
    document.citations = seen

    return document


def _document_body(tex_text: str) -> str:
    """截取 ``\\begin{document}`` 与 ``\\end{document}`` 之间的正文。

    刻意保留原始缩进与空白：下游按字符偏移切分章节，
    任何「顺手做的美化」都会让偏移与原文对不上，导致正文错位。
    """
    start_match = re.search(r"\\begin\{document\}", tex_text)
    if start_match is None:
        return tex_text

    end_match = re.search(r"\\end\{document\}", tex_text)
    end = end_match.start() if end_match else len(tex_text)
    return tex_text[start_match.end():end]


def _extract_preamble(soup, document: LatexDocument, tex_text: str) -> None:
    title_node = soup.find("title")
    if title_node is not None:
        document.title = _clean_latex(_env_text(title_node))

    author_node = soup.find("author")
    if author_node is not None:
        raw = _clean_latex(_strip_author_noise(_env_text(author_node)))
        # 作者之间常用 \and 或逗号分隔
        parts = re.split(r"\s*(?:,|;|、)\s*|\s+and\s+", raw)
        document.authors = [
            p.strip() for p in parts
            if p.strip() and 1 < len(p.strip()) < 60
        ][:50]

    abstract_node = soup.find("abstract")
    if abstract_node is not None:
        document.abstract = _clean_latex(_env_text(abstract_node))
    else:
        _extract_preamble_regex(tex_text, document, only_abstract=True)


def _extract_preamble_regex(tex_text: str, document: LatexDocument, *, only_abstract: bool = False) -> None:
    """正则兜底提取标题与摘要。"""
    if not only_abstract:
        match = re.search(r"\\title\{((?:[^{}]|\{[^{}]*\})*)\}", tex_text)
        if match and not document.title:
            document.title = _clean_latex(match.group(1))

        match = re.search(r"\\author\{((?:[^{}]|\{[^{}]*\})*)\}", tex_text)
        if match and not document.authors:
            raw = _clean_latex(match.group(1))
            document.authors = [
                p.strip() for p in re.split(r"\s*(?:,|and)\s*", raw) if p.strip()
            ][:50]

    if not document.abstract:
        match = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_text, re.S
        )
        if match:
            document.abstract = _clean_latex(match.group(1))


def _extract_bibliography(soup, tex_text: str, document: LatexDocument) -> None:
    """提取 ``\\bibitem`` 条目，用于把引用键映射到具体文献。

    用正则而不是 TexSoup 遍历：``\\bibitem{key}`` 之后的内容是它的**兄弟节点**
    而非子节点，所以按节点取文本只能拿到键名本身。按「下一个 \\bibitem 或
    \\end{thebibliography}」切段才是对的做法。
    """
    pattern = re.compile(
        r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)"
        r"(?=\\bibitem(?:\[[^\]]*\])?\{|\\end\{thebibliography\}|\Z)",
        re.S,
    )
    for match in pattern.finditer(tex_text):
        key = match.group(1).strip()
        if not key:
            continue
        body = _clean_latex(match.group(2))
        document.bibitems[key] = body[:800]


def _walk_sections(text: str) -> list[LatexSection]:
    """按文档顺序遍历章节命令，收集每一节的内容。

    **一遍完成，不做第二遍对齐。** 之前是「先切结构、再按同样的规则切一遍
    取正文」，两遍的输入文本不同（一遍是 document 环境、一遍是全文），
    偏移对不上，结果所有章节的正文都错位到了相邻章节——结论的文字挂在
    方法一节下面，致谢挂在注意力机制下面。这种错误在检索时会表现为
    「引用的位置和内容对不上」，非常难排查。

    现在每个章节的正文在切分时**就地**取到，不存在对齐问题。

    这里不用递归下降而用「按章节命令切段」：TexSoup 的树在遇到自定义宏
    或缺失 ``\\end`` 时会变形，而按命令切段对残缺文档同样有效。
    """
    from TexSoup import TexSoup  # noqa: F401  （保留导入以便外部按需使用）
    # 顺带捕获紧跟标题的 \label —— 论文里很常见（\subsection{Attention}
    # \label{sec:attention}），拿到它才能在别处解析 \ref 的指向
    pattern = re.compile(
        r"\\(section|subsection|subsubsection|paragraph)\*?"
        r"(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}"
        r"\s*(\\label\{([^}]*)\})?"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    level_map = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
    sections: list[LatexSection] = []
    parents: dict[int, str] = {}

    for index, match in enumerate(matches):
        command = match.group(1)
        title = _clean_latex(match.group(2))
        level = level_map.get(command, 1)
        section_label = match.group(4)

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[start:end]

        parents[level] = title
        for deeper in [k for k in parents if k > level]:
            del parents[deeper]
        path = " > ".join(parents[k] for k in sorted(parents))

        section = LatexSection(title=title, level=level, path=path, label=section_label)
        # 正文**就地**取到，不再做第二遍对齐——见函数文档
        section.raw_body = chunk

        _collect_figures(chunk, section)
        _collect_by_regex(chunk, section)

        sections.append(section)

    return sections


def render_tabular(raw: str) -> str:
    """把 ``\\begin{tabular}...`` 渲染成可读的文本表格。

    **为什么必须做这一步。** 论文里几乎没有比表格更浓缩的信息——SOTA 对比、
    消融、超参、复杂度，全在表里。而抽取管线此前只留 ``\\caption{}``、
    把 tabular 正文整个丢掉：实测 320 个「table」分块 100% 只是表注，
    44 篇论文里的 481 个 tabular 环境全部落空。

    后果在笔记里直接可见：85 篇笔记共 385 处 ``⚠️ 待核``，其中 **79% 写的是
    「具体数值未在提供的文本中给出」**——不是论文没写，是我们没给模型看。
    「实验结果」小节里该有数字的地方是一个个 ⚠️。

    输出用 Markdown 表格的形态（``|`` 分隔），模型对这种结构最熟。
    不做对齐、不猜列宽——保留原始的行列关系即可，排版是渲染端的事。
    """
    if not raw:
        return ""

    text = raw

    # 1) 取出所有 tabular 环境（含 tabular*、带列格式参数的）
    #
    # 列格式参数里**会嵌套花括号**：`\begin{tabular}{@{}ll cc@{}c}` 这种写法
    # 用 `\{[^{}]*\}` 匹配不掉，那串 `@ll cc@` 就会当成第一行内容漏出来。
    # 所以要允许一层嵌套。
    bodies = re.findall(
        r"\\begin\{tabular\*?\}(?:\{(?:[^{}]|\{[^{}]*\})*\})?(.*?)\\end\{tabular\*?\}",
        text,
        re.S,
    )
    if not bodies:
        return ""

    rows_out: list[str] = []
    for body in bodies:
        # 2) 去掉只影响排版、不含信息的命令。
        #    \cmidrule 后面可能跟 (r)/(l) 这类裁剪参数，也要一起吃掉，
        #    否则会留下一行 `(r)3-6 (l)7-10`。
        body = re.sub(
            r"\\(?:hline|toprule|midrule|bottomrule|cline|cmidrule|"
            r"addlinespace|smallskip|medskip|bigskip|centering|small|"
            r"footnotesize|scriptsize|tiny|arraybackslash)\b"
            r"(?:\([^)]*\))?(\[[^\]]*\])?(\{[^{}]*\})?",
            "",
            body,
        )
        body = re.sub(r"\\vspace\{[^}]*\}|\\vskip\s*[-\d.]+\w*", "", body)
        body = re.sub(r"\\rule(?:\[[^\]]*\])?\{[^}]*\}\{[^}]*\}", "", body)

        # 3) 按行切。`\\` 是换行，注意它可能是 `\\[2pt]` 形式
        for row in re.split(r"\\\\", body):
            row = row.strip()
            if not row:
                continue

            # 4) \multicolumn{n}{fmt}{text} 只保留 text，列数信息对阅读无意义
            row = re.sub(
                r"\\multicolumn\{[^}]*\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                r"\1",
                row,
            )
            # \multirow 同理，保留内容
            row = re.sub(
                r"\\multirow\{[^}]*\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                r"\1",
                row,
            )

            cells = [_clean_latex(c) for c in row.split("&")]
            cells = [c.strip() for c in cells]
            if not any(cells):
                continue
            rows_out.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows_out)


def _collect_figures(chunk: str, section: LatexSection) -> None:
    """从章节片段里收集图与表。

    用正则而不是 TexSoup 的 ``find_all``：这里处理的是一段文本切片，
    再构造一次 TexSoup 既慢又可能因为片段不完整（缺少配对的 ``\\end``）
    而解析失败。图表的边界在 LaTeX 里很规整，正则足够。
    """
    for kind in ("figure", "table"):
        for match in re.finditer(
            rf"\\begin\{{{kind}\*?\}}(.*?)\\end\{{{kind}\*?\}}", chunk, re.S
        ):
            raw = match.group(1)
            caption_match = re.search(
                r"\\caption(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}", raw, re.S
            )
            label_match = re.search(r"\\label\{([^}]*)\}", raw)
            section.figures.append(
                LatexFigure(
                    kind=kind,
                    caption=_clean_latex(caption_match.group(1)) if caption_match else "",
                    label=label_match.group(1).strip() if label_match else None,
                    graphics=[
                        g.strip()
                        for g in re.findall(
                            r"\\includegraphics(?:\[[^\]]*\])?\*?\{([^}]+)\}", raw
                        )
                        if g.strip()
                    ],
                    # 表格正文。图没有「正文」可抽（内容在图片里），
                    # 只有表才有——而数值恰恰都在表里。
                    body=render_tabular(raw) if kind == "table" else "",
                )
            )




def _make_figure(node, kind: str, raw: str) -> LatexFigure:
    caption = ""
    label = None
    graphics: list[str] = []

    caption_node = node.find("caption")
    if caption_node is not None:
        caption = _clean_latex(_env_text(caption_node))

    label_node = node.find("label")
    if label_node is not None:
        label = _env_text(label_node).strip()

    # 用正则而不是 TexSoup 的 .args 取文件名：`\includegraphics[scale=0.6]{fig}`
    # 里那个可选参数 `[scale=0.6]` 也会出现在 .args 里，被当成文件名收进来。
    # 正则直接跳过方括号部分，干净得多。
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\*?\{([^}]+)\}", raw)
    graphics = [g.strip() for g in graphics if g.strip()]

    return LatexFigure(
        kind=kind,
        caption=caption,
        label=label,
        graphics=graphics,
        body=render_tabular(raw) if kind == "table" else "",
    )


def _collect_by_regex(chunk: str, section: LatexSection) -> None:
    """正则收集公式与引用。

    不依赖 TexSoup 的成功与否——这两类信息对下游（公式渲染、引用图谱）
    很重要，宁可多跑一遍正则也不要漏。
    """
    # 带编号的公式环境
    for match in re.finditer(
        r"\\begin\{(equation|eqnarray|align|gather|multline)\*?\}(.*?)\\end\{\1\*?\}",
        chunk,
        re.S,
    ):
        body = match.group(2)

        # \label 有两个常见位置：环境内部，或紧跟在 \end{equation} 之后。
        # 两种都要找——只找内部的话，相当一部分论文的公式标签会全部丢失
        # （实测 Attention 那篇就是写在环境外面的）。
        label_match = re.search(r"\\label\{([^}]*)\}", body)
        if label_match is None:
            tail = chunk[match.end(): match.end() + 200]
            label_match = re.match(r"\s*\\label\{([^}]*)\}", tail)

        latex = re.sub(r"\\label\{[^}]*\}", "", body).strip()
        if latex:
            section.equations.append(
                LatexEquation(
                    latex=latex,
                    label=label_match.group(1) if label_match else None,
                    numbered="*" not in match.group(0)[:20],
                )
            )

    # 行间公式 \[ \] 与 $$ $$
    for match in re.finditer(r"\\\[(.*?)\\\]|\$\$(.*?)\$\$", chunk, re.S):
        latex = (match.group(1) or match.group(2) or "").strip()
        if latex:
            section.equations.append(LatexEquation(latex=latex, numbered=False))

    # 引用键
    for match in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", chunk):
        for key in match.group(1).split(","):
            key = key.strip()
            if key and key not in section.citations:
                section.citations.append(key)


def _sections_by_regex(tex_text: str) -> list[LatexSection]:
    """纯正则的章节提取，作为 TexSoup 完全失败时的最后兜底。"""
    pattern = re.compile(
        r"\\(section|subsection|subsubsection)\*?"
        r"(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}"
    )
    matches = list(pattern.finditer(tex_text))
    if not matches:
        return []

    level_map = {"section": 1, "subsection": 2, "subsubsection": 3}
    sections: list[LatexSection] = []
    parents: dict[int, str] = {}

    for index, match in enumerate(matches):
        level = level_map.get(match.group(1), 1)
        title = _clean_latex(match.group(2))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tex_text)

        parents[level] = title
        for deeper in [k for k in parents if k > level]:
            del parents[deeper]

        section = LatexSection(
            title=title, level=level,
            path=" > ".join(parents[k] for k in sorted(parents)),
        )
        _collect_by_regex(tex_text[start:end], section)
        sections.append(section)

    return sections


# --------------------------------------------------------------------------
# 正文抽取
# --------------------------------------------------------------------------


def extract_section_text(raw: str) -> list[str]:
    """从章节原始 LaTeX 里切出段落。

    先按空行分段（LaTeX 里空行就是段落分隔），再对每段做转换。
    比按 TexSoup 树遍历更稳——自定义宏、注释、条件编译都不会打乱它。
    """
    # 注释已在 parse_latex 里统一剥过，这里不重复处理

    # 先整段移除图表环境。它们的标题会被单独提取成 figure 块，
    # 留在正文里会让同一条信息出现两次，而且「The Transformer - model
    # architecture.」这种图注混进散文段落里读起来毫无上下文。
    cleaned = re.sub(
        r"\\begin\{(figure|table|algorithm|listing)\*?\}.*?\\end\{\1\*?\}",
        "",
        raw,
        flags=re.S,
    )

    # \includegraphics 会被 pylatexenc 转成 <graphics> 这种占位符，
    # 而且因为字母间距问题会渲染成 "< g r a p h i c s >"，必须显式去掉
    cleaned = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\*?\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"\\(?:begin|end)\{(?:minipage|subfigure|subfloat)\}(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", cleaned)

    # 去掉章节命令本身。子章节在遍历时已经是独立的一节了，如果标题还留在
    # 父节的正文里，就会变成 "§.§ Encoder and Decoder Stacks" 这种噪音——
    # pylatexenc 把 \subsection 转成了 "§.§"，读起来毫无意义。
    cleaned = re.sub(
        r"\\(?:sub)*section\*?(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\\paragraph\*?(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}", "", cleaned)

    # 去掉不产生正文内容的排版命令
    cleaned = re.sub(
        r"\\(?:label|index|vspace|hspace|medskip|smallskip|bigskip|noindent|"
        r"centering|footnotesize|scriptsize|small|large|Large|LARGE|huge|"
        r"raggedright|setlength|addtolength|thispagestyle|pagestyle|"
        r"bibliographystyle|makeatletter|makeatother)\b"
        r"(?:\[[^\]]*\])?(?:\{[^}]*\})?",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\\(?:begin|end)\{(?:center|itemize|enumerate|description)\}", "", cleaned)

    blocks = re.split(r"\n\s*\n", cleaned)
    paragraphs: list[str] = []
    for block in blocks:
        text = _clean_latex(block)
        # 太短的块通常是残留的命令或空白
        if text and len(text) >= 30:
            paragraphs.append(text)
        elif text and paragraphs and len(text) >= 10:
            paragraphs[-1] = f"{paragraphs[-1]} {text}"
    return paragraphs


__all__ = [
    "ARXIV_EPRINT",
    "LatexDocument",
    "LatexEquation",
    "LatexError",
    "LatexFigure",
    "LatexSection",
    "extract_section_text",
    "fetch_arxiv_source",
    "find_main_tex",
    "parse_latex",
    "render_tabular",
]
