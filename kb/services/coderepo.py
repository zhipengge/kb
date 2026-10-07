"""关联与获取论文对应的开源代码。

**顺序很重要**：先从论文正文里找作者自己给的项目链接，找不到才去 GitHub 搜。

正文里的链接是作者声明的，几乎不会有错。而按标题搜索会有一堆同名/仿制的
仓库——搜「UniAD」能搜到官方的，也能搜到别人写的「UniAD 复现笔记」。
把复现笔记当成论文的官方实现挂上去，比没有代码更糟：
用户会以为自己看的是官方实现。

克隆用 ``--depth 1``：知识库要的是「代码长什么样」用于与论文对照，
不是完整的提交历史。深度克隆一个活跃仓库可能上百 MB，浅克隆通常几 MB。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .paths import sanitize_filename

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
USER_AGENT = "kb-knowledge-base/0.1"

# 未认证的 GitHub 搜索接口是 10 次/分钟，核心接口 60 次/小时。
# 批量处理时这个限制是主要瓶颈，必须自己控制节奏。
SEARCH_INTERVAL = 6.5
CORE_INTERVAL = 1.0

_last_search = 0.0
_last_core = 0.0

# 论文正文里的代码仓库链接。覆盖 GitHub / GitLab / Gitee。
_REPO_URL = re.compile(
    r"https?://(?:www\.)?(github\.com|gitlab\.com|gitee\.com)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
    re.I,
)

# 这些路径不是仓库本身，是仓库内的页面
_NOT_A_REPO = {"issues", "pull", "blob", "tree", "releases", "wiki", "settings", "projects"}


@dataclass
class RepoCandidate:
    """一个候选代码仓库。"""

    url: str
    full_name: str
    description: str = ""
    stars: int = 0
    source: str = "search"  # paper_text / search / manual
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "full_name": self.full_name,
            "description": self.description,
            "stars": self.stars,
            "source": self.source,
            "confidence": self.confidence,
        }


def _github(path: str, *, kind: str = "core", timeout: float = 25.0):
    """带节流的 GitHub API 调用。"""
    global _last_search, _last_core

    now = time.monotonic()
    if kind == "search":
        wait = SEARCH_INTERVAL - (now - _last_search)
    else:
        wait = CORE_INTERVAL - (now - _last_core)
    if wait > 0:
        time.sleep(wait)

    request = urllib.request.Request(  # noqa: S310 - 前缀是常量
        GITHUB_API + path,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            if kind == "search":
                _last_search = time.monotonic()
            else:
                _last_core = time.monotonic()
            return json.loads(response.read()), None
    except urllib.error.HTTPError as exc:
        if kind == "search":
            _last_search = time.monotonic()
        else:
            _last_core = time.monotonic()
        if exc.code == 403:
            return None, "GitHub 接口限流（未认证时搜索 10 次/分钟）"
        if exc.code == 404:
            return None, "不存在"
        return None, f"HTTP {exc.code}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# 从正文里找
# --------------------------------------------------------------------------


def extract_repo_links(text: str) -> list[str]:
    """从论文文本里提取代码仓库链接。

    返回规范化的 ``https://github.com/owner/repo`` 形式（去掉路径后缀）。
    """
    if not text:
        return []

    seen: list[str] = []
    for match in _REPO_URL.finditer(text):
        host, owner, repo = match.group(1).lower(), match.group(2), match.group(3)
        # 剥掉 .git 与末尾标点
        repo = re.sub(r"\.git$", "", repo).rstrip(".,;)")
        # 排除「仓库内的页面」被误当成仓库名（github.com/foo/blob 之类）
        if owner.lower() in _NOT_A_REPO or repo.lower() in _NOT_A_REPO:
            continue
        url = f"https://{host}/{owner}/{repo}"
        if url not in seen:
            seen.append(url)
    return seen


def _paper_text(paper) -> str:
    """收集论文里可能藏链接的文本。

    优先看 LaTeX 源码——作者通常把项目地址写在脚注或引言里，
    而 PDF 抽取时脚注常被排到页面边缘，容易丢。
    """
    from ..extensions import db
    from ..models import Chunk

    parts: list[str] = [paper.abstract or ""]

    chunks = (
        db.session.query(Chunk.text)
        .filter(Chunk.paper_id == paper.id)
        .order_by(Chunk.ord)
        .limit(120)
        .all()
    )
    parts.extend(row[0] for row in chunks if row[0])

    # 已经下载过源码的话，直接扫源码全文（正则找 URL 很便宜）
    source_dir = (paper.meta or {}).get("source_dir")
    if source_dir:
        directory = Path(source_dir)
        if directory.is_dir():
            for tex in list(directory.rglob("*.tex"))[:20]:
                try:
                    parts.append(tex.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue

    return "\n".join(parts)


# --------------------------------------------------------------------------
# GitHub 搜索
# --------------------------------------------------------------------------


def search_github(title: str, *, limit: int = 5) -> list[RepoCandidate]:
    """按论文标题搜 GitHub 仓库。"""
    query = urllib.parse.quote(f'"{title}"')
    data, error = _github(
        f"/search/repositories?q={query}&sort=stars&order=desc&per_page={limit}",
        kind="search",
    )

    if data is None and "限流" in (error or ""):
        # 短语搜索常常无结果，退一步用普通关键词
        query = urllib.parse.quote(title)
        data, error = _github(
            f"/search/repositories?q={query}&sort=stars&order=desc&per_page={limit}",
            kind="search",
        )

    if data is None or "items" not in data:
        log.debug("GitHub 搜索失败：%s", error)
        return []

    results: list[RepoCandidate] = []
    for item in data["items"]:
        results.append(
            RepoCandidate(
                url=item.get("html_url", ""),
                full_name=item.get("full_name", ""),
                description=item.get("description") or "",
                stars=int(item.get("stargazers_count") or 0),
                source="search",
            )
        )
    return results


def _score_candidate(candidate: RepoCandidate, title: str, arxiv_id: str | None) -> float:
    """给候选仓库打分。

    这不是精确科学，但排序信号很明确：
      * 描述里出现论文标题 —— 强信号（作者通常照抄标题）
      * 描述里出现 arXiv 编号 —— 非常强（唯一标识）
      * 星数 —— 官方实现通常远高于复现
      * 名字里带 "reproduce"、"reimplementation"、"notes" —— 负信号
    """
    from .ingest import similarity

    score = 0.0
    text = f"{candidate.full_name} {candidate.description}".lower()

    if arxiv_id and arxiv_id.lower() in text:
        score += 60
    score += similarity(title, candidate.description) * 0.5

    if candidate.stars >= 500:
        score += 25
    elif candidate.stars >= 100:
        score += 18
    elif candidate.stars >= 20:
        score += 10
    elif candidate.stars >= 5:
        score += 4

    # 复现、笔记、教程类仓库不是官方实现
    for marker in ("reproduc", "re-implement", "reimplement", "notes", "tutorial",
                   "paper reading", "reading notes", "学习", "笔记", "复现"):
        if marker in text:
            score -= 35

    return max(0.0, min(100.0, score))


def find_repo(paper, *, min_score: float = 45.0) -> tuple[RepoCandidate | None, str]:
    """为论文找对应的代码仓库。

    返回 ``(候选, 说明)``。分数不够就返回 None——
    **宁可没有代码，也不要挂错代码**。挂错的代价是用户以为自己在看官方实现。
    """
    # 1) 正文里的链接（最可靠）
    links = extract_repo_links(_paper_text(paper))
    if links:
        owner_repo = "/".join(links[0].split("/")[-2:])
        return (
            RepoCandidate(
                url=links[0],
                full_name=owner_repo,
                source="paper_text",
                confidence=95.0,
                evidence={"matched": "论文正文中的链接", "all_links": links[:5]},
            ),
            f"论文正文里给出了 {links[0]}",
        )

    # 2) GitHub 搜索
    if not paper.title:
        return None, "论文没有标题，无法搜索"

    try:
        candidates = search_github(paper.title)
    except Exception as exc:
        return None, f"搜索失败：{exc}"

    if not candidates:
        return None, "GitHub 上没有找到相关仓库"

    arxiv_id = paper.arxiv_id
    scored = [( _score_candidate(c, paper.title, arxiv_id), c) for c in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]

    if best_score < min_score:
        return None, (
            f"最接近的是 {best.full_name}（{best_score:.0f} 分，低于阈值 {min_score:.0f}）"
            f"，未能确认是官方实现"
        )

    best.confidence = best_score
    best.evidence = {
        "matched": "GitHub 搜索",
        "candidates": [c.to_dict() for _, c in scored[:3]],
    }
    return best, f"匹配到 {best.full_name}（{best_score:.0f} 分，{best.stars} 星）"


# --------------------------------------------------------------------------
# 克隆
# --------------------------------------------------------------------------


def clone_repo(url: str, dest_root: Path, *, name: str | None = None) -> tuple[Path | None, str]:
    """浅克隆仓库到目标目录。

    用 ``--depth 1``：知识库要的是「代码长什么样」用于与论文对照，
    不是提交历史。一个活跃仓库的完整历史可能上百 MB，浅克隆通常几 MB。
    """
    if not shutil.which("git"):
        return None, "系统里没有 git，无法克隆"

    owner_repo = "/".join(url.rstrip("/").split("/")[-2:])
    folder = sanitize_filename(name or owner_repo.replace("/", "__"), fallback="repo")
    target = Path(dest_root) / folder

    if (target / ".git").is_dir():
        return target, "已存在，跳过克隆"

    Path(dest_root).mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(target)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(target, ignore_errors=True)
        return None, "克隆超时（超过 10 分钟）"
    except Exception as exc:
        return None, f"克隆失败：{exc}"

    if result.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        message = (result.stderr or result.stdout or "").strip().splitlines()
        return None, f"克隆失败：{message[-1][:120] if message else '未知错误'}"

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    return target, f"{size / 1024 / 1024:.1f} MB"


def repo_size(path: Path) -> dict:
    """统计仓库构成，用于「论文 ↔ 代码」对照时给个概览。"""
    counts: dict[str, int] = {}
    total = 0
    for item in Path(path).rglob("*"):
        if not item.is_file() or ".git" in item.parts:
            continue
        suffix = item.suffix.lower() or "(无扩展名)"
        counts[suffix] = counts.get(suffix, 0) + 1
        total += 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
    return {"files": total, "by_extension": dict(top)}


def read_repo_readme(path: Path, *, limit: int = 4000) -> str:
    """读仓库的 README。它是判断「这个仓库是什么」最直接的依据。"""
    for candidate in ("README.md", "README.rst", "README.txt", "readme.md", "README"):
        file = Path(path) / candidate
        if file.is_file():
            try:
                return file.read_text(encoding="utf-8", errors="replace")[:limit]
            except OSError:
                continue
    return ""


__all__ = [
    "RepoCandidate",
    "clone_repo",
    "extract_repo_links",
    "find_repo",
    "read_repo_readme",
    "repo_size",
    "search_github",
]
