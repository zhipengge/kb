"""基于数据库的任务队列与进程内 worker。

三个关键机制，缺一不可：

**原子抢占。** 用一条 ``UPDATE ... WHERE id = (SELECT ... LIMIT 1) RETURNING id``
完成「选中 + 占有」。分两步（先 SELECT 再 UPDATE）会让两个 worker 拿到同一个任务——
而这类 bug 在单线程测试里永远不出现，只会在真实并发时冒出来。

**租约。** worker 崩溃时任务会停在 ``running``。``locked_at`` 让这些任务可以被
回收：超过租约时间没有心跳的 running 任务，在下次启动时重新排队。
没有租约，一次断电就会让若干任务永远卡在「运行中」。

**去重键。** 用户在界面上连点两次「重建索引」会入队两个一模一样的任务，
它们跑起来会互相踩。``dedupe_key`` 保证同类任务在未完成时只存在一个。
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from flask import Flask

from ..extensions import db
from ..models import Job, JobEvent
from ..models.base import utcnow

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    """任务被请求取消。在阶段边界抛出，由 worker 捕获并标记为已取消。"""


class JobContext:
    """交给任务函数的上下文：报告进度、写日志、检查取消。

    任务函数**不应该**自己去改 Job 行——把状态变更收拢到这里，
    才能保证进度、日志、租约心跳三者在同一次提交里一致更新。
    """

    def __init__(self, app: Flask, job_id: str):
        self.app = app
        self.job_id = job_id
        self._last_heartbeat = 0.0

    # --- 进度与日志 ---
    def progress(self, value: float, message: str | None = None) -> None:
        """上报进度。``value`` 取 0..1。"""
        value = max(0.0, min(1.0, float(value)))
        with self.app.app_context():
            job = db.session.get(Job, self.job_id)
            if job is None:
                return
            job.progress = value
            if message:
                job.message = message[:512]
            job.locked_at = utcnow()
            db.session.commit()
        if message:
            self.log(message)

    def log(self, message: str, level: str = "info", **data: Any) -> None:
        self.app.logger.log(
            logging.WARNING if level == "warning" else logging.ERROR if level == "error" else logging.INFO,
            "[job %s] %s", self.job_id, message,
        )
        with self.app.app_context():
            db.session.add(
                JobEvent(job_id=self.job_id, level=level, message=message, data=data or None)
            )
            db.session.commit()

    # --- 取消 ---
    def check_cancelled(self) -> None:
        """在阶段边界调用。任务函数应把它放在「可以安全停下」的位置。"""
        with self.app.app_context():
            row = db.session.execute(
                db.select(Job.cancel_requested).where(Job.id == self.job_id)
            ).scalar()
        if row:
            raise JobCancelled()

    @property
    def cancelled(self) -> bool:
        with self.app.app_context():
            return bool(
                db.session.execute(
                    db.select(Job.cancel_requested).where(Job.id == self.job_id)
                ).scalar()
            )

    def heartbeat(self) -> None:
        """续租。耗时长的阶段应定期调用，避免被误判为崩溃。"""
        now = time.monotonic()
        if now - self._last_heartbeat < 15:
            return
        self._last_heartbeat = now
        with self.app.app_context():
            job = db.session.get(Job, self.job_id)
            if job is not None:
                job.locked_at = utcnow()
                db.session.commit()


# --------------------------------------------------------------------------
# 入队
# --------------------------------------------------------------------------


def enqueue(
    job_type: str,
    params: dict | None = None,
    *,
    priority: int = 100,
    dedupe_key: str | None = None,
    max_attempts: int = 3,
    paper_id: str | None = None,
) -> Job:
    """把一个任务放进队列。

    ``dedupe_key`` 命中未完成任务时直接返回那个任务，不再入队新任务——
    这是幂等的关键。API 里带 ``Idempotency-Key`` 的请求最终落到这里。
    """
    if dedupe_key:
        existing = (
            db.session.query(Job)
            .filter(Job.dedupe_key == dedupe_key, Job.status.in_(("queued", "running")))
            .order_by(Job.created_at.desc())
            .first()
        )
        if existing is not None:
            log.debug("任务去重命中：%s -> %s", dedupe_key, existing.id)
            return existing

    job = Job(
        type=job_type,
        params=params or {},
        priority=priority,
        dedupe_key=dedupe_key,
        max_attempts=max_attempts,
        paper_id=paper_id,
        status="queued",
    )
    db.session.add(job)
    db.session.commit()
    log.info("任务入队：%s %s", job.type, job.id)
    return job


# --------------------------------------------------------------------------
# 抢占
# --------------------------------------------------------------------------


def claim_next(worker_id: str) -> Job | None:
    """原子地取出并占有一个待执行任务。没有则返回 None。"""
    row = db.session.execute(
        db.text(
            """
            UPDATE jobs
               SET status = 'running',
                   locked_at = :now,
                   started_at = COALESCE(started_at, :now),
                   attempts = attempts + 1,
                   worker = :worker
             WHERE id = (
                   SELECT id FROM jobs
                    WHERE status = 'queued'
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
             )
            RETURNING id
            """
        ),
        {"now": utcnow(), "worker": worker_id},
    ).fetchone()
    db.session.commit()

    if row is None:
        return None
    return db.session.get(Job, row[0])


def recover_stale_jobs(lease_seconds: int) -> int:
    """回收租约过期的 running 任务。

    进程被杀、机器断电、容器被重启都会留下这类任务。恢复策略是重新排队，
    而不是直接标记失败——大多数任务是幂等的（扫描、解析、建索引），
    重跑一遍没有任何副作用。
    """
    cutoff = utcnow() - timedelta(seconds=lease_seconds)
    rows = db.session.execute(
        db.text(
            """
            UPDATE jobs
               SET status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'queued' END,
                   error = CASE WHEN attempts >= max_attempts
                                THEN '任务在多次尝试后仍未完成（可能是进程中断）'
                                ELSE error END,
                   locked_at = NULL,
                   worker = NULL,
                   finished_at = CASE WHEN attempts >= max_attempts THEN :now ELSE NULL END
             WHERE status = 'running'
               AND (locked_at IS NULL OR locked_at < :cutoff)
            RETURNING id, attempts, max_attempts
            """
        ),
        {"cutoff": cutoff, "now": utcnow()},
    ).fetchall()
    db.session.commit()

    for job_id, attempts, max_attempts in rows:
        requeued = attempts < max_attempts
        log.warning(
            "回收僵死任务 %s（第 %d/%d 次）-> %s",
            job_id, attempts, max_attempts, "重新排队" if requeued else "标记失败",
        )
        db.session.add(
            JobEvent(
                job_id=job_id,
                level="warning",
                message="任务被中断，已重新排队" if requeued else "任务多次中断，已放弃",
            )
        )
    if rows:
        db.session.commit()
    return len(rows)


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


class Worker:
    """进程内 worker：若干个线程轮询队列并执行任务。"""

    def __init__(self, app: Flask, concurrency: int = 2, poll_interval: float = 1.0):
        self.app = app
        self.concurrency = max(1, concurrency)
        self.poll_interval = poll_interval
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- 生命周期 ---
    def start(self) -> None:
        lease = self._lease_seconds()
        with self.app.app_context():
            recovered = recover_stale_jobs(lease)
        if recovered:
            self.app.logger.warning("启动时回收了 %d 个僵死任务", recovered)

        for index in range(self.concurrency):
            thread = threading.Thread(
                target=self._loop, name=f"kb-worker-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        self.app.logger.info("worker 已启动：%d 个线程（%s）", self.concurrency, self.worker_id)

    def stop(self) -> None:
        self._stop.set()

    def _lease_seconds(self) -> int:

        return int(self.app.extensions["kb_settings"].get("jobs.lease_seconds"))

    # --- 主循环 ---
    def _loop(self) -> None:
        from .tasks import TASKS

        while not self._stop.is_set():
            job = None
            try:
                with self.app.app_context():
                    job = claim_next(self.worker_id)
                if job is None:
                    self._stop.wait(self.poll_interval)
                    continue
                self._run_one(job, TASKS)
            except Exception:
                # worker 线程绝不能因为单个任务的问题而死掉——
                # 死掉之后任务会静默堆积，界面上的表现是「点了没反应」。
                self.app.logger.exception("worker 线程异常")
                self._stop.wait(self.poll_interval)
                continue

    def _run_one(self, job: Job, tasks: dict) -> None:
        ctx = JobContext(self.app, job.id)
        started = time.perf_counter()

        handler: Callable | None = tasks.get(job.type)
        if handler is None:
            self._finish_failed(job.id, f"未知的任务类型：{job.type}")
            return

        ctx.log(f"开始执行 {job.type}")
        try:
            with self.app.app_context():
                result = handler(ctx, job.params or {})
            elapsed = int((time.perf_counter() - started) * 1000)
            with self.app.app_context():
                row = db.session.get(Job, job.id)
                if row is not None:
                    row.status = "succeeded"
                    row.progress = 1.0
                    row.result = result if isinstance(result, dict) else {"value": result}
                    row.finished_at = utcnow()
                    row.locked_at = None
                    db.session.commit()
            ctx.log(f"完成，用时 {elapsed} ms")

        except JobCancelled:
            with self.app.app_context():
                row = db.session.get(Job, job.id)
                if row is not None:
                    row.status = "cancelled"
                    row.finished_at = utcnow()
                    row.locked_at = None
                    db.session.commit()
            ctx.log("任务已取消", level="warning")

        except Exception as exc:
            self.app.logger.exception("任务 %s 执行失败", job.id)
            self._finish_failed(job.id, str(exc), ctx=ctx)

    def _finish_failed(self, job_id: str, error: str, ctx: JobContext | None = None) -> None:
        with self.app.app_context():
            row = db.session.get(Job, job_id)
            if row is None:
                return
            if row.attempts < row.max_attempts:
                # 还有重试机会，退回队列而不是直接失败。
                # 网络抖动、模型限流这类错误重试一次通常就好了。
                row.status = "queued"
                row.error = error
                row.locked_at = None
                row.worker = None
                db.session.commit()
                if ctx:
                    ctx.log(f"失败（将在第 {row.attempts + 1} 次重试）：{error}", level="warning")
                return
            row.status = "failed"
            row.error = error
            row.finished_at = utcnow()
            row.locked_at = None
            db.session.commit()
        if ctx:
            ctx.log(f"最终失败：{error}", level="error")


_worker: Worker | None = None
_worker_lock = threading.Lock()


def start_embedded_worker(app: Flask) -> Worker | None:
    """在 Web 进程内启动 worker（幂等）。"""
    global _worker
    with _worker_lock:
        if _worker is not None:
            return _worker
        if app.config.get("TESTING"):
            return None
        concurrency = int(app.extensions["kb_settings"].get("jobs.concurrency"))
        _worker = Worker(app, concurrency=concurrency)
        _worker.start()
        return _worker


def get_embedded_worker() -> Worker | None:
    return _worker


__all__ = [
    "JobCancelled",
    "JobContext",
    "Worker",
    "claim_next",
    "enqueue",
    "get_embedded_worker",
    "recover_stale_jobs",
    "start_embedded_worker",
]
