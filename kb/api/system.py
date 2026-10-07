"""系统类接口：健康检查、配置读写、任务查询。"""

from __future__ import annotations

import json
import time

from flask import Response, current_app, request

from ..extensions import db
from ..models import Job, JobEvent
from ..utils.time import iso
from . import api_bp
from .auth import actor_name, require_scope
from .envelope import error_response, ok, paged, parse_paging


@api_bp.get("/system/ping")
def ping():
    """存活探测。**不需要鉴权**——监控系统不该为了探活而持有凭据。"""
    return ok({"pong": True, "time": iso(_now())})


@api_bp.get("/system/health")
@require_scope("read")
def health():
    """完整健康报告：数据库能力、各项统计、模型配置状态。

    这是 agent 接入后第一个该调的接口——它能一次性回答
    「这个知识库现在能不能用、里面有什么、能不能做向量检索」。
    """
    from ..models import Chunk, Note, Paper, Tag

    report = current_app.extensions.get("kb_preflight")
    vec_state = current_app.extensions.get("kb_vector_state", {})
    settings = current_app.extensions["kb_settings"]
    cfg = current_app.extensions["kb_boot_config"]

    counts = {
        "papers": db.session.query(Paper).filter(Paper.deleted_at.is_(None)).count(),
        "notes": db.session.query(Note).count(),
        "chunks": db.session.query(Chunk).count(),
        "tags": db.session.query(Tag).count(),
    }
    active_jobs = (
        db.session.query(Job).filter(Job.status.in_(("queued", "running"))).count()
    )

    llm_provider = settings.get("llm.provider")
    llm_ready = bool(
        settings.get_secret("llm.api_key") if llm_provider == "anthropic"
        else settings.get_secret("llm.api_key") or settings.get("llm.base_url")
    )

    return ok(
        {
            "version": current_app.config.get("KB_VERSION"),
            "database": report.to_dict() if report else None,
            "vector": {
                "enabled": bool(vec_state.get("enabled")),
                "version": vec_state.get("version"),
                "model": settings.get("embedding.model") or None,
            },
            "counts": counts,
            "jobs": {"active": active_jobs},
            "llm": {
                "provider": llm_provider,
                "deep_model": settings.get("llm.deep_model"),
                "configured": llm_ready,
            },
            "paths": {
                "papers_roots": settings.papers_roots,
                "notes_root": settings.notes_root,
                "codes_root": settings.codes_root,
            },
            "data_dir": str(cfg.data_dir),
        }
    )


def _now():
    from ..models.base import utcnow

    return utcnow()


# --------------------------------------------------------------------------
# 设置
# --------------------------------------------------------------------------


@api_bp.get("/settings")
@require_scope("read")
def get_settings():
    """读取运行时设置。密钥一律返回掩码。"""
    settings = current_app.extensions["kb_settings"]
    from ..settings import GROUPS

    return ok({"values": settings.all_with_source(), "groups": GROUPS})


@api_bp.patch("/settings")
@require_scope("admin")
def patch_settings():
    """批量修改设置。

    任一字段校验失败则整批不生效——半生效的配置比失败的配置更难排查。
    """
    from ..settings import SettingsError

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return error_response("invalid_argument", "请求体需要是一个 JSON 对象")

    values = payload.get("values", payload)
    if not isinstance(values, dict):
        return error_response("invalid_argument", "values 字段需要是一个 JSON 对象")

    settings = current_app.extensions["kb_settings"]
    try:
        updated = settings.update_many(values, updated_by=actor_name())
    except SettingsError as exc:
        return error_response("invalid_argument", str(exc))

    return ok({"values": updated})


# --------------------------------------------------------------------------
# 任务
# --------------------------------------------------------------------------


def _job_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "priority": job.priority,
        "progress": job.progress,
        "message": job.message,
        "params": job.params,
        "result": job.result,
        "error": job.error,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "paper_id": job.paper_id,
        "created_at": iso(job.created_at),
        "started_at": iso(job.started_at),
        "finished_at": iso(job.finished_at),
    }


@api_bp.get("/jobs")
@require_scope("read")
def list_jobs():
    limit, cursor = parse_paging()
    status = request.args.get("status")

    query = db.session.query(Job)
    if status:
        query = query.filter(Job.status == status)
    if cursor:
        query = query.filter(Job.id < cursor)

    rows = query.order_by(Job.id.desc()).limit(limit + 1).all()
    has_more = len(rows) > limit
    rows = rows[:limit]

    return paged(
        [_job_dict(j) for j in rows],
        next_cursor=rows[-1].id if has_more and rows else None,
        limit=limit,
    )


@api_bp.get("/jobs/<job_id>")
@require_scope("read")
def get_job(job_id: str):
    job = db.session.get(Job, job_id)
    if job is None:
        return error_response("not_found", "任务不存在", 404)
    return ok(_job_dict(job))


@api_bp.post("/jobs/<job_id>/cancel")
@require_scope("write")
def cancel_job(job_id: str):
    job = db.session.get(Job, job_id)
    if job is None:
        return error_response("not_found", "任务不存在", 404)
    if job.is_terminal:
        return error_response("conflict", f"任务已经结束（{job.status}），无法取消", 409)

    job.cancel_requested = True
    db.session.commit()
    return ok({"id": job.id, "cancel_requested": True})


@api_bp.get("/jobs/<job_id>/events")
@require_scope("read")
def job_events(job_id: str):
    """任务日志的 SSE 流。

    用 SSE 而不是 WebSocket：进度推送是单向的，SSE 在 Flask 里就是
    一个普通响应，不需要额外协议升级，也不需要给 gunicorn 换 worker 类型。
    """
    job = db.session.get(Job, job_id)
    if job is None:
        return error_response("not_found", "任务不存在", 404)

    after = request.args.get("after", type=float) or 0.0

    def generate():
        last_ts = after
        idle_rounds = 0
        while True:
            events = (
                db.session.query(JobEvent)
                .filter(JobEvent.job_id == job_id)
                .order_by(JobEvent.ts)
                .all()
            )
            for event in events:
                stamp = event.ts.timestamp()
                if stamp <= last_ts:
                    continue
                last_ts = stamp
                payload = {
                    "id": event.id,
                    "ts": iso(event.ts),
                    "level": event.level,
                    "message": event.message,
                    "data": event.data,
                }
                yield f"event: log\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

            db.session.expire_all()
            fresh = db.session.get(Job, job_id)
            if fresh is None:
                break
            yield (
                "event: progress\n"
                f"data: {json.dumps({'progress': fresh.progress, 'status': fresh.status, 'message': fresh.message}, ensure_ascii=False)}\n\n"
            )
            if fresh.is_terminal:
                yield (
                    "event: done\n"
                    f"data: {json.dumps(_job_dict(fresh), ensure_ascii=False)}\n\n"
                )
                break

            # 长时间没有新事件时发心跳，避免代理断掉空闲连接
            idle_rounds += 1
            if idle_rounds % 15 == 0:
                yield ": keepalive\n\n"
            time.sleep(0.8)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 让 nginx 不要缓冲 SSE
            "Connection": "keep-alive",
        },
    )
