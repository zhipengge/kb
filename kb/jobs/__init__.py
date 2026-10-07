"""后台任务。

单机部署不引入 Redis/Celery：``jobs`` 表就是队列，进程内线程池就是 worker。
好处是零外部依赖、任务状态天然持久化、重启不丢；代价是并发上限受单机限制，
且多进程部署时要显式指定「只有一个进程跑 worker」。
"""

from .queue import JobCancelled, JobContext, enqueue, start_embedded_worker

__all__ = ["JobCancelled", "JobContext", "enqueue", "start_embedded_worker"]
