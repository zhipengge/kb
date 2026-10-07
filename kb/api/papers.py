"""论文接口。

设计上有一条贯穿始终的约束：**接口只接受 id，不接受路径**。

调用方永远不需要、也不应该知道论文在磁盘上的位置。所有文件访问都通过
``/papers/{id}/file`` 流式返回，路径从未离开服务端。这样路径穿越在结构上
就不可能发生——不是靠校验输入来防，而是根本没有接受路径的入口。
"""

from __future__ import annotations

import logging

from flask import request, send_file

# 序列化函数定义在服务层，MCP 工具与 REST 共用同一份。
# 这里转出来，好让 kb/api/tags.py 那句 `from .papers import paper_dict` 继续可用。
from ..services.papers import paper_dict
from . import api_bp
from .auth import require_scope
from .envelope import error_response, ok, paged, parse_paging

log = logging.getLogger(__name__)

__all__ = ["paper_dict"]


@api_bp.get("/papers")
@require_scope("read")
def list_papers_endpoint():
    """论文列表。

    支持按关键词、标签、年份、会议、阅读状态、是否有代码筛选。
    ``q`` 同时匹配标题、摘要、作者、DOI 与 arXiv 号。
    """
    from ..services import papers as papers_service

    limit, cursor = parse_paging()
    tag_ids = request.args.getlist("tag")

    rows, next_cursor, total = papers_service.list_papers(
        query=request.args.get("q"),
        tag_ids=tag_ids or None,
        year=request.args.get("year", type=int),
        venue=request.args.get("venue"),
        reading_status=request.args.get("reading_status"),
        ingest_status=request.args.get("ingest_status"),
        has_repo=request.args.get("has_code", type=lambda v: v.lower() in {"1", "true", "yes"}),
        include_deleted=request.args.get("include_deleted", type=lambda v: v.lower() == "true"),
        sort=request.args.get("sort", "added"),
        limit=limit,
        cursor=cursor,
    )

    return paged(
        [paper_dict(p) for p in rows],
        next_cursor=next_cursor,
        limit=limit,
        total=total,
    )


@api_bp.get("/papers/<paper_id>")
@require_scope("read")
def get_paper_endpoint(paper_id: str):
    from ..services import papers as papers_service

    paper = papers_service.get_paper(paper_id, include_deleted=True)
    if paper is None:
        return error_response("not_found", "论文不存在", 404)
    return ok(paper_dict(paper, detail=True))


@api_bp.get("/papers/<paper_id>/text")
@require_scope("read")
def get_paper_text(paper_id: str):
    """取论文正文分块，带定位信息。

    「检索 → 读原文 → 引用」里的第二步。检索只给回几个片段，想回答
    「这个方法具体怎么做」得能往下读。三种读法：

      * 不传参数——返回**全篇目录**加开头几块，先看清结构；
      * ``?section=Method``——按小节名子串匹配；
      * ``?page=4``——取该页正文，用来核对某句话是否真在第 N 页。

    每块都带 ``locator``（形如 ``§3 Method p.4``），可直接当出处。
    """
    from ..services import papers as papers_service

    page = request.args.get("page", type=int)
    result = papers_service.paper_text(
        paper_id,
        page=page,
        section=request.args.get("section") or None,
        offset=request.args.get("offset", 0, type=int) or 0,
        limit=request.args.get("limit", 12, type=int) or 12,
    )
    if result.get("error"):
        return error_response("not_found", result["error"], 404)
    return ok(result)


@api_bp.patch("/papers/<paper_id>")
@require_scope("write")
def patch_paper_endpoint(paper_id: str):
    """更新论文元数据。"""
    from ..services import papers as papers_service

    payload = request.get_json(silent=True) or {}
    try:
        paper = papers_service.update_paper(paper_id, payload)
    except papers_service.PaperError as exc:
        return error_response("invalid_argument", str(exc), 400)
    return ok(paper_dict(paper, detail=True))


@api_bp.delete("/papers/<paper_id>")
@require_scope("write")
def delete_paper_endpoint(paper_id: str):
    """软删除论文。

    默认只标记删除（文件和笔记都留着）。``?purge=true`` 才会真正清除，
    这是不可逆操作，需要 admin 权限。
    """
    from ..services import papers as papers_service
    from .auth import has_scope

    purge = request.args.get("purge", type=lambda v: v.lower() == "true")
    try:
        if purge:
            if not has_scope("admin"):
                return error_response("forbidden", "彻底删除需要 admin 权限", 403)
            papers_service.purge(paper_id)
            return ok({"id": paper_id, "purged": True})
        paper = papers_service.soft_delete(paper_id)
    except papers_service.PaperError as exc:
        return error_response("not_found", str(exc), 404)
    return ok(paper_dict(paper))


@api_bp.get("/papers/<paper_id>/file")
@require_scope("read")
def get_paper_file(paper_id: str):
    """流式返回 PDF 本体。

    ``send_file`` 会处理 Range 请求，因此 PDF.js 这类阅读器可以直接拖动
    进度条而不必下载整个文件。这也是不自己读文件再返回的原因。
    """
    from ..services import papers as papers_service

    paper = papers_service.get_paper(paper_id)
    if paper is None:
        return error_response("not_found", "论文不存在", 404)

    import os

    if not paper.file_path or not os.path.isfile(paper.file_path):
        return error_response("file_missing", "论文文件已不在磁盘上", 410)

    return send_file(
        paper.file_path,
        mimetype="application/pdf",
        as_attachment=request.args.get("download", type=lambda v: v.lower() == "true"),
        download_name=f"{(paper.title or paper.id)[:80]}.pdf",
        conditional=True,  # 支持 Range / If-Modified-Since
    )


@api_bp.post("/papers/<paper_id>/reading-status")
@require_scope("write")
def set_reading_status(paper_id: str):
    from ..services import papers as papers_service

    payload = request.get_json(silent=True) or {}
    status = payload.get("status")
    try:
        paper = papers_service.mark_reading(paper_id, status)
    except papers_service.PaperError as exc:
        return error_response("invalid_argument", str(exc), 400)
    return ok({"id": paper.id, "reading_status": paper.reading_status})


@api_bp.get("/system/papers-stats")
@require_scope("read")
def papers_stats():
    """论文库的分组统计。

    挂在 ``/system`` 下而不是 ``/papers/stats``：后者会和 ``/papers/{id}``
    长得一样，虽然路由优先级能正确区分，但读代码的人要多想一步。
    """
    from ..services import papers as papers_service

    return ok(papers_service.stats())
