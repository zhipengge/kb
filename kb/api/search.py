"""检索接口。

对外提供两种粒度：``/search`` 返回**分块**（给需要精确引用的调用方），
``/search/papers`` 返回**论文**（给「找几篇相关的来看看」这类需求）。

返回的每条结果都带 ``chunk_id`` 与 ``locator``（页码/章节）。这两样东西合起来
构成了 agent 的核心工作流：先搜到片段，判断相关性，需要更多上下文时
用 ``chunk_id`` 回溯，最后带着可验证的定位作答。
"""

from __future__ import annotations

from flask import request

from . import api_bp
from .auth import require_scope
from .envelope import error_response, ok, paged, parse_paging


def _filters_from_request() -> dict:
    filters: dict = {}
    for key in ("paper_id", "note_id", "kind"):
        value = request.args.get(key) or (request.get_json(silent=True) or {}).get(key)
        if value:
            filters[key] = value

    paper_ids = request.args.getlist("paper_ids")
    if paper_ids:
        filters["paper_ids"] = paper_ids

    # 标签筛选：先解析成论文 id 集合
    tag_ids = request.args.getlist("tag")
    if tag_ids:
        from ..extensions import db
        from ..models import PaperTag

        rows = (
            db.session.query(PaperTag.paper_id)
            .filter(PaperTag.tag_id.in_(tag_ids))
            .distinct()
            .all()
        )
        ids = [row[0] for row in rows]
        if not ids:
            return {"__empty__": True}
        filters["paper_ids"] = ids

    return filters


def _search_options() -> tuple[str, int, bool, int]:
    """从设置与查询串里取出检索参数。

    查询串可以覆盖设置里的值——调用方临时想要更多结果时不必先去改配置。
    """
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    mode = request.args.get("mode") or "hybrid"
    if mode not in {"hybrid", "fts", "vector"}:
        mode = "hybrid"
    limit = request.args.get("limit", type=int) or int(settings.get("retrieval.top_k"))
    limit = max(1, min(limit, 100))
    group_by_paper = request.args.get("group_by_paper", type=lambda v: v.lower() == "true")
    if group_by_paper is None:
        group_by_paper = bool(settings.get("retrieval.group_by_paper"))
    rrf_k = int(settings.get("retrieval.rrf_k"))
    return mode, limit, group_by_paper, rrf_k


@api_bp.get("/search")
@require_scope("read")
def search_endpoint():
    """检索分块。

    支持 ``q``、``mode``、``limit``、``paper_id``、``tag``、``group_by_paper``。
    也接受 POST，便于把复杂筛选条件放在请求体里。
    """
    return _do_search()


@api_bp.post("/search")
@require_scope("read")
def search_post_endpoint():
    return _do_search()


def _do_search():
    from ..services.search import search

    payload = request.get_json(silent=True) or {}
    # 接受 q / query 两个字段名。对外接口的字段名要保守地用「大家都会猜的那个」：
    # 实测拿 query 来调会被拒（提示「缺少查询词 q」），而隔壁 /ask 是收
    # question 的——同一套接口里两种叫法，调用方只能靠试。
    # 多认一个别名不会让接口变模糊，却能让猜错的调用方直接成功。
    query = (
        request.args.get("q")
        or request.args.get("query")
        or payload.get("q")
        or payload.get("query")
        or ""
    )
    if not str(query).strip():
        return error_response(
            "invalid_argument", "缺少查询词：请提供 q（或 query）", 400
        )

    filters = _filters_from_request()
    if filters.pop("__empty__", False):
        return paged([], next_cursor=None, limit=0, total=0)

    mode, limit, group_by_paper, rrf_k = _search_options()

    hits = search(
        query,
        limit=limit,
        mode=mode,
        filters=filters,
        rrf_k=rrf_k,
        group_by_paper=group_by_paper,
    )

    return ok(
        [hit.to_dict() for hit in hits],
        meta={"query": query, "mode": mode, "count": len(hits), "filters": filters},
    )


@api_bp.get("/search/papers")
@require_scope("read")
def search_papers_endpoint():
    """检索论文（按论文聚合）。"""
    from ..services.search import search_papers

    query = request.args.get("q") or ""
    if not query.strip():
        return error_response("invalid_argument", "缺少查询词 q", 400)

    filters = _filters_from_request()
    if filters.pop("__empty__", False):
        return ok([], meta={"query": query, "count": 0})

    limit, _cursor = parse_paging()
    results = search_papers(query, limit=min(limit, 50), filters=filters)
    return ok(results, meta={"query": query, "count": len(results)})


@api_bp.get("/search/similar/<chunk_id>")
@require_scope("read")
def similar_chunks(chunk_id: str):
    """找与某个分块相似的其它分块。

    用途是「读过这一段，还想看看别处怎么说的」。没有向量检索时退化为
    用该块的文本做一次全文检索——效果差一些但仍然是可用的。
    """
    from ..extensions import db
    from ..models import Chunk
    from ..services.search import search

    chunk = db.session.get(Chunk, chunk_id)
    if chunk is None:
        return error_response("not_found", "分块不存在", 404)

    hits = search(chunk.text[:400], limit=6, filters={"paper_id": None})
    results = [hit.to_dict() for hit in hits if hit.chunk_id != chunk_id][:5]
    return ok(results, meta={"source_chunk_id": chunk_id, "count": len(results)})


@api_bp.get("/search/stats")
@require_scope("read")
def search_stats():
    """索引与检索能力状态。

    一次性回答调用方最关心的几个问题：索引建到哪了、向量检索能不能用、
    如果不能用是什么原因。
    """
    from flask import current_app

    from ..services.embedding import embedding_stats
    from ..services.indexer import index_stats

    return ok(
        {
            "index": index_stats(),
            "embedding": embedding_stats(),
            "fts": current_app.extensions.get("kb_fts_state", {}),
            "vector": current_app.extensions.get("kb_vector_state", {}),
        }
    )
