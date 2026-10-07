"""路径校验与安全解析。

**这是整个系统里唯一允许把用户输入拼进文件路径的地方。** 其它模块要用文件系统，
一律调用这里。把这条规则守住，路径穿越就不可能发生——散落各处的
``os.path.join`` 才是这类漏洞的来源。

威胁模型里有两类「用户输入」：
  1. 网页/接口里填的根目录；
  2. **模型生成的路径**——聊天 Agent 调用工具时，参数是模型产出的，
     和陌生人输入没有区别。schema 校验只保证它是字符串，不保证它不是 ``/etc/shadow``。

另外这里处理了一批 Windows/WSL 特有的坑：保留文件名（CON、PRN、NUL…）、
结尾的点与空格、大小写不敏感的文件系统。这些在 Linux 上写代码时完全想不到，
但用户的论文目录就在 Windows 盘上。
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

# 不允许作为根目录的路径。注意判断的是「归一化后的绝对路径」，
# 所以 /mnt 被拒但 /mnt/papers 允许——前者是挂载点，后者才是用户的文献目录。
# ruff: noqa: S108 —— 这里的 "/tmp" 是**禁止**被设为根目录的路径之一，
# 不是在使用临时文件。bandit 的这条规则在这份数据上不适用。
_DENY_ROOTS = {
    "/", "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/root",
    "/run", "/sbin", "/sys", "/tmp", "/usr", "/var", "/home", "/mnt", "/media",
    "/opt", "/srv",
}
_MIN_DEPTH = 2  # 至少两级，挡掉 /mnt、/data 这类过宽的根

# Windows 保留设备名。这些名字在任何扩展名下都非法：
# CON.pdf、con.txt 在 Windows 上都无法创建/打开。
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_MAX_NAME_LEN = 120
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class PathError(ValueError):
    """路径不合法。消息面向用户，会直接显示在设置页或接口错误里。"""


# --------------------------------------------------------------------------
# 根目录校验
# --------------------------------------------------------------------------


def normalize(path: str | os.PathLike[str]) -> Path:
    """展开 ~ 与变量，转成绝对路径。不解析符号链接（那要看具体用途）。"""
    return Path(os.path.expandvars(str(path))).expanduser().absolute()


def validate_root(path: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    """校验一个「根目录」是否可以作为文献库的根。

    规则故意保守：宁可让用户多填一级目录，也不要允许把 ``/`` 或
    ``/mnt`` 设成根——那会让扫描器遍历整块盘，也会让路径白名单形同虚设。
    """
    if not str(path).strip():
        raise PathError("路径不能为空")

    resolved = normalize(path)

    # 逐级解析已存在的部分，识别符号链接指向
    try:
        resolved = resolved.resolve()
    except OSError as exc:
        raise PathError(f"无法解析路径 {path}：{exc}") from exc

    posix = resolved.as_posix().rstrip("/") or "/"

    if posix in _DENY_ROOTS:
        raise PathError(
            f"不能把 {posix} 设为根目录——它包含系统或其它用户的数据。"
            "请指向具体的文献目录，例如 /mnt/papers。"
        )

    parts = [p for p in posix.split("/") if p]
    if len(parts) < _MIN_DEPTH:
        raise PathError(
            f"根目录至少要 {_MIN_DEPTH} 级，{posix} 过于宽泛。"
            "过宽的根目录会让扫描器遍历大量无关文件。"
        )

    # 家目录本身也不适合当根（下面通常混着各种东西）
    home = Path.home().resolve()
    if resolved == home:
        raise PathError(f"不能把家目录 {home} 本身设为根目录，请指向它下面的某个子目录")

    if must_exist and not resolved.is_dir():
        raise PathError(f"目录不存在：{resolved}")

    return resolved


def check_roots(roots: list[str] | None) -> list[dict]:
    """检查一组根目录的状态，供界面展示。

    不抛异常——这个函数存在的意义正是**在界面上如实呈现问题**，
    而不是替用户决定能不能继续。
    """
    result = []
    for raw in roots or []:
        item: dict = {"path": raw, "exists": False, "is_dir": False, "readable": False,
                      "writable": False, "pdf_count": 0, "error": None}
        if not raw or not str(raw).strip():
            item["error"] = "空路径"
            result.append(item)
            continue
        try:
            resolved = normalize(raw)
        except (OSError, ValueError) as exc:
            item["error"] = str(exc)
            result.append(item)
            continue

        item["path"] = str(resolved)
        try:
            item["exists"] = resolved.exists()
            item["is_dir"] = resolved.is_dir()
            if item["is_dir"]:
                item["readable"] = os.access(resolved, os.R_OK)
                item["writable"] = os.access(resolved, os.W_OK)
        except OSError as exc:
            item["error"] = str(exc)
        result.append(item)
    return result


def ensure_default_dirs(settings) -> tuple[list[str], list[str]]:
    """创建缺失的默认目录。返回 ``(已创建, 错误信息)``。"""
    created: list[str] = []
    errors: list[str] = []

    targets = [*settings.papers_roots, settings.notes_root, settings.codes_root]
    for raw in targets:
        if not raw or not str(raw).strip():
            continue
        try:
            path = normalize(raw)
        except (OSError, ValueError) as exc:
            errors.append(f"{raw}：{exc}")
            continue
        if path.is_dir():
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path))
        except OSError as exc:
            errors.append(f"无法创建 {path}：{exc}")
    return created, errors


# --------------------------------------------------------------------------
# 路径解析（防穿越）
# --------------------------------------------------------------------------


def resolve_within(
    root: str | os.PathLike[str],
    relative: str | os.PathLike[str],
    *,
    must_exist: bool = False,
) -> Path:
    """把相对路径解析到 root 之内，越界则报错。

    关键点是 ``resolve()``：它会展开符号链接，所以「根目录里放一个指向
    /etc 的软链」这种绕过方式会在这里被同一个包含判断拦住——
    如果只用字符串前缀比较（``startswith``），这类绕过就防不住。
    """
    root_path = Path(root)
    try:
        root_resolved = root_path.resolve()
    except OSError as exc:
        raise PathError(f"根目录不可用：{root_path}") from exc

    candidate = Path(relative)
    if candidate.is_absolute():
        raise PathError(f"这里需要相对路径：{relative}")

    # 显式拒绝 .. 段：即使 resolve 之后仍在根内，「用 .. 跳出去再跳回来」
    # 也是不该允许的写法，它会掩盖真正的意图
    if ".." in candidate.parts:
        raise PathError("路径中不允许出现 ..")

    target = (root_resolved / candidate).resolve()

    if not _is_within(target, root_resolved):
        raise PathError(f"路径越界：{relative} 不在 {root_resolved} 之内")

    if must_exist and not target.exists():
        raise PathError(f"文件不存在：{relative}")

    return target


def is_safe_path(path: str | os.PathLike[str], roots: list[str]) -> bool:
    """判断一个绝对路径是否落在任一已注册的根目录内。"""
    try:
        target = Path(path).resolve()
    except OSError:
        return False
    for root in roots:
        try:
            if _is_within(target, Path(root).resolve()):
                return True
        except OSError:
            continue
    return False


def _is_within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def path_key(path: str | os.PathLike[str]) -> str:
    """归一化的路径键，用于在大小写不敏感的文件系统上识别同一个文件。

    Windows 盘（以及 WSL 挂载的 /mnt/*）不区分大小写，
    ``Paper.pdf`` 和 ``paper.PDF`` 是同一个文件。只按原始路径比较会得出两条记录，
    进而在同一个文件上重复建索引。
    """
    text = os.path.normpath(str(path)).replace("\\", "/")
    return os.path.normcase(text).lower()


def shorten_for_fs(path: str | os.PathLike[str], max_total: int = 240) -> Path:
    """把过长的路径截短，避免撞上 Windows 的 260 字符上限。

    Windows 的 MAX_PATH 限制在名字很长（论文标题当文件名很常见）+ 嵌套目录时
    很容易碰到。截的是中间，保留开头和结尾——结尾通常带扩展名，要留住。
    """
    text = str(path)
    if len(text) <= max_total:
        return Path(text)

    parent = os.path.dirname(text)
    stem, ext = os.path.splitext(os.path.basename(text))
    keep = max_total - len(parent) - len(ext) - 2
    if keep < 8:
        # 父目录本身就太长了，只能硬截
        return Path(text[:max_total])
    return Path(parent) / f"{stem[:keep]}~{ext}"


# --------------------------------------------------------------------------
# 文件名清洗
# --------------------------------------------------------------------------


def sanitize_filename(name: str, *, fallback: str = "untitled", max_len: int = _MAX_NAME_LEN) -> str:
    """把任意字符串变成安全的文件名。

    处理 Windows 的三种坑：非法字符、保留设备名、结尾的点与空格
    （``foo.`` 和 ``foo `` 在 Windows 上创建后会变成 ``foo``，
    导致「写进去的名字」和「读出来的名字」不一致）。
    """
    if not name:
        return fallback

    # 归一化 Unicode：不同来源的「é」可能是组合字符也可能是单码点，
    # 不归一化会让视觉相同的两个名字产生两个文件
    text = unicodedata.normalize("NFC", str(name))

    text = _UNSAFE_CHARS.sub("_", text)
    text = text.replace("../", "_").replace("..\\", "_")
    text = text.strip()

    # 结尾的点与空格（Windows 会静默丢弃）
    text = text.rstrip(". ")

    # 折叠空白
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return fallback

    # Windows 保留名：即使带扩展名也不行，所以比对主干部分
    stem = text.split(".")[0].lower()
    if stem in _WINDOWS_RESERVED:
        text = f"_{text}"

    if len(text) > max_len:
        stem, ext = os.path.splitext(text)
        keep = max_len - len(ext)
        text = f"{stem[:max(1, keep)]}{ext}"

    return text or fallback


def slugify(text: str, *, fallback: str = "item", max_len: int = 80) -> str:
    """生成 URL / 文件名友好的短标识。

    保留 CJK 字符——论文标题常常是中文，把它们全部剔除会得到一堆 ``untitled``，
    反而失去了可读性。真正需要剔除的只是路径分隔符与控制字符。
    """
    if not text:
        return fallback
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = _UNSAFE_CHARS.sub("", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-.")
    text = text.rstrip(". ")
    if not text:
        return fallback
    return text[:max_len].rstrip("-.") or fallback


def is_excluded(name: str, patterns: list[str] | None) -> bool:
    """判断文件名/目录名是否命中排除规则。"""
    for pattern in patterns or []:
        if not pattern:
            continue
        if fnmatch.fnmatch(name, pattern):
            return True
        # 允许 ``.*`` 这种写法匹配所有隐藏项
        if pattern.startswith(".") and name.startswith("."):
            return True
    return False


__all__ = [
    "PathError",
    "check_roots",
    "ensure_default_dirs",
    "is_excluded",
    "is_safe_path",
    "normalize",
    "path_key",
    "resolve_within",
    "sanitize_filename",
    "shorten_for_fs",
    "slugify",
    "validate_root",
]
