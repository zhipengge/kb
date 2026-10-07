"""把关联的开源代码索引进检索层。

**为什么需要。** 代码仓库一直关联在论文上（59 个仓库、51 个有 README、
语言统计和本地目录），但它们只存在于 ``code_repos`` 表里，**从没进过检索层**。
后果是问「这篇论文的代码在哪里」时，模型只能引用正文里恰好出现的
GitHub 链接——而很多论文没在正文里给 URL，或者 URL 藏在脚注里。
用户拿到的是「资料里没有相关信息」，可数据库里明明躺着仓库地址。

索引的内容刻意**不做**整个仓库的源码：

  * 全量代码索引的体量会淹没论文正文（一个仓库动辄上千文件），
    检索结果被代码淹没，反而更难找到论文里的方法描述；
  * 常见问题是「代码在哪、怎么跑、用了什么框架」，README + 依赖文件
    就能回答；
  * 「某个函数怎么实现的」需要的是代码检索而非文本检索（按符号、按调用图），
    那是另一个量级的工作，不该假装用全文检索能解决。

所以每个仓库索引成两类块：一段概述（含地址、语言、README）和一段
顶层结构（目录与关键文件），二者都带 ``kind=code`` 以便按类型过滤。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..extensions import db
from ..models import Chunk, CodeRepo
from ..models.chunk import CHUNK_CODE
from .chunker import rules_meta

log = logging.getLogger(__name__)

# 依赖/配置文件——回答「怎么装、依赖什么」靠它们
DEPENDENCY_FILES = (
    "requirements.txt", "environment.yml", "environment.yaml",
    "setup.py", "pyproject.toml", "Pipfile", "package.json",
    "CMakeLists.txt", "conda.yaml",
)

# 顶层目录里出现这些名字说明是元数据而非代码，列表里跳过
_NOISE_DIRS = frozenset({
    ".git", ".github", ".idea", ".vscode", "__pycache__", "node_modules",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".mypy_cache",
})

MAX_TREE_ENTRIES = 60


def _repo_summary(repo: CodeRepo) -> str:
    """仓库概述块：地址 + 语言 + README。"""
    lines = [f"# 代码仓库：{repo.name}", ""]
    if repo.url:
        lines.append(f"仓库地址：{repo.url}")
    if repo.local_path:
        lines.append(f"本地路径：{repo.local_path}")
    if repo.vcs:
        lines.append(f"版本控制：{repo.vcs}")
    if repo.head_commit:
        lines.append(f"当前提交：{repo.head_commit[:12]}")

    stats = repo.language_stats or {}
    if isinstance(stats, dict) and stats:
        # 语言统计有两种可能的形状（{lang: bytes} 或 {languages: {...}}），都兼容
        table = stats.get("languages") if isinstance(stats.get("languages"), dict) else stats
        pairs = [
            (str(k), v) for k, v in table.items()
            if isinstance(v, (int, float)) and not str(k).startswith("_")
        ]
        if pairs:
            total = sum(v for _, v in pairs) or 1
            top = sorted(pairs, key=lambda kv: -kv[1])[:6]
            rendered = "、".join(f"{k} {v / total * 100:.0f}%" for k, v in top)
            lines.append(f"主要语言：{rendered}")

    if repo.readme_excerpt:
        lines.append("")
        lines.append("## README 摘要")
        lines.append(repo.readme_excerpt.strip())

    return "\n".join(lines)


def _repo_tree(repo: CodeRepo) -> str | None:
    """顶层结构 + 依赖文件内容。取不到就返回 None。"""
    if not repo.local_path:
        return None
    root = Path(repo.local_path)
    if not root.is_dir():
        return None

    entries: list[str] = []
    try:
        children = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return None

    for child in children:
        if child.name in _NOISE_DIRS or child.name.startswith("."):
            continue
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{child.name}{suffix}")
        if len(entries) >= MAX_TREE_ENTRIES:
            break

    if not entries:
        return None

    lines = [f"# 代码结构：{repo.name}", "", "顶层内容：", *[f"- {e}" for e in entries]]

    # 依赖文件的内容往往直接回答「怎么跑起来」
    for name in DEPENDENCY_FILES:
        path = root / name
        if not path.is_file():
            continue
        try:
            content = path.read_text(errors="replace")[:1500].strip()
        except OSError:
            continue
        if content:
            lines.append("")
            lines.append(f"## {name}")
            lines.append("```")
            lines.append(content)
            lines.append("```")

    return "\n".join(lines)


def index_code_repos(*, force: bool = False) -> dict[str, Any]:
    """为所有关联了仓库的论文建立代码索引块。

    幂等：重跑会先删掉该仓库已有的代码块再重建，不会累积重复。
    """
    repos = (
        db.session.query(CodeRepo)
        .filter(CodeRepo.paper_id.isnot(None))
        .all()
    )

    created = skipped = 0
    for repo in repos:
        if not force:
            existing = (
                db.session.query(Chunk)
                .filter(Chunk.paper_id == repo.paper_id, Chunk.kind == CHUNK_CODE)
                .count()
            )
            if existing:
                skipped += 1
                continue

        # 先清掉旧的，保证重跑不累积
        db.session.query(Chunk).filter(
            Chunk.paper_id == repo.paper_id, Chunk.kind == CHUNK_CODE
        ).delete(synchronize_session=False)

        blocks: list[tuple[str, str]] = [
            (f"代码仓库 {repo.name}", _repo_summary(repo)),
        ]
        tree = _repo_tree(repo)
        if tree:
            blocks.append((f"代码结构 {repo.name}", tree))

        for order, (section, text) in enumerate(blocks):
            if not text.strip():
                continue
            db.session.add(
                Chunk(
                    paper_id=repo.paper_id,
                    kind=CHUNK_CODE,
                    section_path=section,
                    ord=order,
                    text=text,
                    n_tokens=len(text) // 3,
                    meta={"repo": repo.name, "url": repo.url, **rules_meta()},
                )
            )
            created += 1

    db.session.commit()
    log.info("代码索引完成：新增 %d 块，跳过 %d 个已有仓库", created, skipped)
    return {"chunks": created, "repos": len(repos), "skipped": skipped}


__all__ = ["index_code_repos"]
