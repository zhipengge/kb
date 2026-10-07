"""系统域模型：设置、任务队列、流水线产物、API Key、审计日志。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, IdMixin, TimestampMixin, utcnow

# --------------------------------------------------------------------------
# 设置
# --------------------------------------------------------------------------


class Setting(IdMixin, TimestampMixin, Base):
    """网页端可改的运行时配置。

    值和 schema 分开：这里只存值，字段的类型/默认值/含义由 ``kb.settings``
    里的声明式定义负责。这样加一个新配置项只需要改一处，
    而且数据库里的脏值可以在读取时被 schema 兜住。
    """

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    value: Mapped[dict | list | str | int | float | bool | None] = mapped_column(JSON)
    updated_by: Mapped[str | None] = mapped_column(String(64))


# --------------------------------------------------------------------------
# 任务队列
# --------------------------------------------------------------------------

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
JOB_STATUSES = (JOB_QUEUED, JOB_RUNNING, JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED)

TERMINAL_STATUSES = (JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED)


class Job(IdMixin, TimestampMixin, Base):
    """一个后台任务。这张表同时就是队列本身（没有 Redis/Celery）。

    抢占靠 ``UPDATE ... SET status='running', locked_at=... WHERE id = (SELECT ...
    WHERE status='queued' ...) RETURNING id`` —— 单条语句完成「选中并占有」，
    不会出现两个 worker 拿到同一个任务。`UPDATE ... RETURNING` 需要 SQLite >= 3.35，
    本机是 3.46，已实测可用。

    ``locked_at`` 是租约：worker 崩溃后不会永远占着任务，超过租约时间的
    running 任务在下次启动时会被回收重新排队。
    """

    __tablename__ = "jobs"

    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    params: Mapped[dict | None] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=JOB_QUEUED, index=True
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)  # 小的先跑

    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # 0..1
    message: Mapped[str | None] = mapped_column(String(512))

    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    worker: Mapped[str | None] = mapped_column(String(64))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    cancel_requested: Mapped[bool] = mapped_column(default=False)

    # 去重键：同一个 key 的未完成任务不会重复入队
    # （比如连点两次「重建索引」，或扫描任务被并发触发）
    dedupe_key: Mapped[str | None] = mapped_column(String(255), index=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关联对象，便于 UI 直接跳转
    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="SET NULL"), index=True
    )

    __table_args__ = (
        # 队列取任务的查询模式：status + priority + created_at
        Index("ix_jobs_queue", "status", "priority", "created_at"),
        Index("ix_jobs_dedupe", "dedupe_key", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Job {self.id} {self.type} {self.status}>"

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class JobEvent(IdMixin, Base):
    """任务的日志行。同时是 SSE 推送的数据源。

    单独的（而不是复用一家日志库的）原因是：这些事件要展示在网页上、
    要通过 API 暴露给外部 agent，还要在任务结束后保留以便排查。
    """

    __tablename__ = "job_events"

    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    data: Mapped[dict | None] = mapped_column(JSON)

    job: Mapped[Job] = relationship()


class Artifact(IdMixin, TimestampMixin, Base):
    """深度阅读流水线的阶段产物。

    流水线被切成若干阶段（抽取 → 结构化 → 分节精读 → 综合 → 打标 → 建索引 …），
    每个阶段的结果都存一份，并用 ``fingerprint`` 标记它的「输入条件」：
    论文内容哈希 + 阶段名 + 提示词版本 + 模型 + 参数。

    指纹没变就直接复用，于是「改一条提示词」只会让受影响的阶段重跑，
    而不是整条链从头再来——这在调试提示词时是数量级的差别。
    """

    __tablename__ = "artifacts"

    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # 产物既可能是文件（图、大 JSON），也可能是内联的小 JSON
    path: Mapped[str | None] = mapped_column(String(2048))
    payload: Mapped[dict | list | None] = mapped_column(JSON)

    job_id: Mapped[str | None] = mapped_column(String(26))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("paper_id", "stage", "fingerprint", name="uq_artifact_fingerprint"),
    )


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------

# 权限范围
SCOPE_READ = "read"
SCOPE_WRITE = "write"      # 增改笔记、标签
SCOPE_INGEST = "ingest"    # 新增论文、触发扫描与重建索引
SCOPE_ADMIN = "admin"      # 改配置、管密钥
SCOPES = (SCOPE_READ, SCOPE_WRITE, SCOPE_INGEST, SCOPE_ADMIN)


class ApiKey(IdMixin, TimestampMixin, Base):
    """对外接口的凭据。

    只存哈希，明文仅在创建时返回一次——这是唯一安全的做法，
    因为「找回 API Key」这个需求本身就不该被满足（该重新签发）。
    """

    __tablename__ = "api_keys"

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    scopes: Mapped[list | None] = mapped_column(JSON, default=list)
    rate_limit: Mapped[str | None] = mapped_column(String(32))  # 形如 "120/minute"

    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ApiKey {self.name} {self.prefix}…>"

    @property
    def is_active(self) -> bool:
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and self.expires_at < utcnow():
            return False
        return True


class AuditLog(IdMixin, Base):
    """对外接口的调用审计。

    外部 agent 能写东西进知识库，所以「谁在什么时候改了什么」必须有据可查。
    """

    __tablename__ = "audit_log"

    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    actor: Mapped[str | None] = mapped_column(String(128))  # key:名字 / web / cli
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target: Mapped[str | None] = mapped_column(String(255))
    method: Mapped[str | None] = mapped_column(String(8))
    path: Mapped[str | None] = mapped_column(String(512))
    status_code: Mapped[int | None] = mapped_column(Integer)
    ip: Mapped[str | None] = mapped_column(String(64))
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)


__all__ = [
    "JOB_FAILED",
    "JOB_QUEUED",
    "JOB_RUNNING",
    "JOB_STATUSES",
    "JOB_SUCCEEDED",
    "SCOPES",
    "SCOPE_ADMIN",
    "SCOPE_INGEST",
    "SCOPE_READ",
    "SCOPE_WRITE",
    "TERMINAL_STATUSES",
    "ApiKey",
    "Artifact",
    "AuditLog",
    "Job",
    "JobEvent",
    "LLMUsage",
    "Setting",
    "WebSearchCache",
]


# --------------------------------------------------------------------------
# 模型花费账本
# --------------------------------------------------------------------------


class LLMUsage(IdMixin, TimestampMixin, Base):
    """一次模型调用的用量记录。

    **为什么要有这个。** 这套系统里花钱的地方不少（精读、打标、查询扩展、
    联网检索的判断），而一笔笔调用散在各处、谁也不知道累计花了多少。
    实测一次全库重生成跑掉 132 万 token，事前事后都只能靠猜。

    只记 **token 数**，不记金额：token 是可核对的事实，价格是会变的配置。
    折算成钱放在读取侧做（见 services/budget.py），这样调价不会让历史账目
    失真——一条记着「$0.23」的旧记录，在换模型之后就没有意义了。

    ``kind`` 说明这笔钱花在哪个环节，用于回答「钱花在哪了」：
    read=精读、tag=打标、expand=查询扩展、chat=对话、other=其它。
    """

    __tablename__ = "llm_usage"

    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="other", index=True)
    model: Mapped[str | None] = mapped_column(String(128))
    # 关联对象（论文/会话），出问题时能顺着查回去
    ref: Mapped[str | None] = mapped_column(String(64), index=True)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # 本次调用是否被预算闸门拦下（拦下的不产生费用，但要留痕）
    blocked: Mapped[bool] = mapped_column(default=False)

    __table_args__ = (Index("ix_llm_usage_kind_created", "kind", "created_at"),)


# --------------------------------------------------------------------------
# 联网检索缓存
# --------------------------------------------------------------------------


class WebSearchCache(IdMixin, TimestampMixin, Base):
    """一次联网检索的完整结果。

    **为什么值得缓存。** 三个原因，都是实测出来的：

    1. 这些接口本身就抖——实测 OpenAlex 同一小时内先返回 200、后超时，
       StackExchange 三次里失败一次。缓存让「某个源正在抖的时候」
       仍然有结果可给。
    2. 抓正文（``fetch_pages``）是整个链路最慢的一步，一次一页 HTTP +
       trafilatura 抽取。而 agent 通过 MCP 调 ``search`` 时**重复查询很常见**。
    3. 联网结果几分钟内不会变，重复打网络纯属浪费。

    存的是**完整结果**（含 ``content`` 与 ``extra``），不是给前端看的那几个字段——
    见 ``services/websearch.py`` 里 ``_serialize`` 的说明。
    """

    __tablename__ = "web_search_cache"

    # query + kinds + limit + fetch_pages 的哈希。不把这些都算进去会串味：
    # 同一个问句用不同 kinds 得到的是完全不同的结果。
    key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    payload: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # 注意：SQLite **不存时区**，写进去的 aware datetime 读回来是 naive 的，
    # 而存的确实是 UTC 值（不转本地时）。所以比较时不能直接和 utcnow() 比，
    # 要用 services/websearch.py 里的 _as_naive_utc() 统一口径——
    # 直接比会抛 TypeError: can't compare offset-naive and offset-aware datetimes，
    # 而那个异常会被缓存的容错逻辑吞掉，表现为「缓存永远不命中、却查不出错」。
    # timezone=True 在 SQLite 上是空操作，加了也没用。
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)

    __table_args__ = (UniqueConstraint("key", name="uq_web_search_cache_key"),)
