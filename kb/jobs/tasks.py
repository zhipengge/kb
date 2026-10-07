"""任务类型注册表。

每种后台任务在这里注册一个处理函数，签名为 ``(ctx: JobContext, params: dict) -> dict``。

处理函数内部一律用**惰性导入**引入服务层：任务模块会在 app 启动时就被加载，
如果在这里顶层导入所有服务，任何一处导入失败都会让整个应用起不来。
惰性导入把失败限制在实际执行那个任务的时候。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from flask import current_app

from ..extensions import db
from .queue import JobContext

log = logging.getLogger(__name__)


def task_ping(ctx: JobContext, params: dict) -> dict:
    """连通性自检任务。

    存在的意义是让「队列到底能不能跑」这件事可以被独立验证——
    排查任务系统问题时，你不希望同时还要怀疑具体业务逻辑。
    """
    steps = int(params.get("steps", 3))
    delay = float(params.get("delay", 0.2))
    for i in range(steps):
        ctx.check_cancelled()
        time.sleep(delay)
        ctx.progress((i + 1) / steps, f"第 {i + 1}/{steps} 步")
    return {"steps": steps, "ok": True}


def task_scan(ctx: JobContext, params: dict) -> dict:
    """扫描论文目录并入库。"""
    from ..services.scanner import scan_roots

    return scan_roots(ctx, roots=params.get("roots"), full=bool(params.get("full")))


def task_index(ctx: JobContext, params: dict) -> dict:
    """解析 PDF、分块、建全文索引。"""
    from ..services.indexer import index_papers

    return index_papers(ctx, paper_ids=params.get("paper_ids"), force=bool(params.get("force")))


def task_embed(ctx: JobContext, params: dict) -> dict:
    """为分块计算向量并写入向量表。"""
    from ..services.embedding import embed_pending

    return embed_pending(ctx, paper_ids=params.get("paper_ids"), force=bool(params.get("force")))


def task_read(ctx: JobContext, params: dict) -> dict:
    """对一篇论文跑深度阅读流水线，产出笔记。"""
    from ..models import Paper
    from ..services.budget import BudgetExceeded
    from ..services.reading import run_pipeline

    paper_ids = params.get("paper_ids") or ([params["paper_id"]] if params.get("paper_id") else [])
    if not paper_ids:
        raise ValueError("没有指定要精读的论文")

    results = []
    total = len(paper_ids)
    for index, paper_id in enumerate(paper_ids):
        ctx.check_cancelled()
        ctx.progress(index / total, f"精读 {index + 1}/{total}")
        paper = db.session.get(Paper, paper_id)
        if paper is None:
            results.append({"paper_id": paper_id, "error": "论文不存在"})
            continue
        try:
            result = run_pipeline(paper, ctx=ctx)
            results.append(result.to_dict())
        except BudgetExceeded as exc:
            # **超预算要立刻停，不是逐篇失败下去。**
            # 预算是个全局闸门，剩下的每一篇都会撞同一堵墙——继续跑只会
            # 产出几十条一模一样的错误，把真正的失败原因淹掉，
            # 还要白等一轮。已经做完的那些结果照常返回，不浪费。
            log.warning("精读因超出预算中止：%s", exc)
            results.append({"paper_id": paper_id, "error": str(exc), "aborted": True})
            return {
                "total": total,
                "completed": len([r for r in results if "error" not in r]),
                "results": results,
                "aborted": str(exc),
            }
        except Exception as exc:
            log.exception("精读论文 %s 失败", paper_id)
            results.append({"paper_id": paper_id, "error": str(exc)})

    return {"total": total, "results": results}


def task_ingest(ctx: JobContext, params: dict) -> dict:
    """按 arXiv 编号下载论文并入库。

    为什么做成任务而不是同步请求：解析 → 下载 PDF → 解析正文建索引，
    合起来要十几到几十秒。放在请求里做，浏览器会先超时，而这期间
    用户看不到任何反馈。做成任务后进度走 job_events，界面能显示到哪一步了。

    **解析和下载分开是有意的**：解析（标题/编号 → arXiv 条目）很快，
    做成同步让用户在下载之前就看到「找到的是哪一篇」并确认。
    标题匹配偶尔会认错，让人确认一次比事后发现库里混进了别的论文便宜得多。
    """
    from ..services import papers as papers_service
    from ..services.ingest import download_pdf
    from ..services.paths import ensure_default_dirs

    arxiv_id = (params.get("arxiv_id") or "").strip()
    title = (params.get("title") or "").strip()
    if not arxiv_id:
        raise ValueError("没有指定 arXiv 编号")

    settings = current_app.extensions["kb_settings"]
    roots = settings.papers_roots
    if not roots:
        raise ValueError("没有配置论文根目录，先在设置里指定")

    _created, errors = ensure_default_dirs(settings)
    for message in errors:
        log.warning("目录创建：%s", message)

    target_dir = Path(roots[0]) / "_arxiv"
    target_dir.mkdir(parents=True, exist_ok=True)

    ctx.progress(0.15, f"下载 PDF（{arxiv_id}）")
    path, message = download_pdf(
        arxiv_id,
        target_dir,
        filename=f"{arxiv_id.replace('.', '_')}_{(title or arxiv_id)[:60]}.pdf",
    )
    if path is None:
        raise ValueError(f"下载失败：{message}")

    ctx.check_cancelled()
    ctx.progress(0.55, "解析正文并建立索引")
    paper = papers_service.create_from_path(
        str(path), source="arxiv", title=title or None
    )

    # 补上 arXiv 侧的元数据——比从 PDF 首页抽的准
    meta = dict(paper.meta or {})
    meta.update(
        {
            "arxiv_id": arxiv_id,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "ingested_from": "web",
        }
    )
    paper.arxiv_id = arxiv_id
    paper.meta = meta
    db.session.commit()

    return {"paper_id": paper.id, "title": paper.title, "arxiv_id": arxiv_id}


TASKS: dict[str, Callable[[JobContext, dict], Any]] = {
    "ping": task_ping,
    "read": task_read,
    "ingest": task_ingest,
    "scan": task_scan,
    "index": task_index,
    "embed": task_embed,
}

__all__ = ["TASKS", "task_embed", "task_index", "task_ping", "task_scan"]
