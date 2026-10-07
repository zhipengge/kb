"""文献域模型：论文、重复候选、扫描记录、代码仓库关联。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

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

# 关系目标用字符串声明，由 SQLAlchemy 注册表在 mapper 配置期解析。
# 这里只在类型检查时导入，运行期不导入——否则 paper/note/chunk/tag 之间
# 会形成循环导入。
if TYPE_CHECKING:
    from .chunk import Chunk
    from .note import Note
    from .tag import PaperTag

# --------------------------------------------------------------------------
# 枚举取值（存字符串而非数据库枚举：SQLite 的 CHECK 约束改起来要重建表）
# --------------------------------------------------------------------------

SOURCE_LOCAL = "local"      # 从磁盘扫描发现
SOURCE_UPLOAD = "upload"    # 网页/接口上传
SOURCE_URL = "url"          # 从链接抓取
SOURCE_DOI = "doi"
SOURCE_ARXIV = "arxiv"
SOURCES = (SOURCE_LOCAL, SOURCE_UPLOAD, SOURCE_URL, SOURCE_DOI, SOURCE_ARXIV)

INGEST_PENDING = "pending"    # 只有文件，还没解析
INGEST_PARSED = "parsed"      # 已抽取正文
INGEST_INDEXED = "indexed"    # 已分块并建索引
INGEST_FAILED = "failed"
INGEST_STATUSES = (INGEST_PENDING, INGEST_PARSED, INGEST_INDEXED, INGEST_FAILED)

READING_UNREAD = "unread"
READING_READING = "reading"
READING_READ = "read"
READING_REVIEWED = "reviewed"
READING_STATUSES = (READING_UNREAD, READING_READING, READING_READ, READING_REVIEWED)


class Paper(IdMixin, TimestampMixin, Base):
    """一篇论文。

    关于「文件身份」的设计：论文的物理载体是磁盘上的一个 PDF，用户可能移动、
    重命名它，也可能在多个根目录下各放一份。这里用三个字段共同描述身份：

      * ``file_path`` —— 当前位置（绝对路径）。
      * ``path_key``  —— 归一化后的路径（大小写折叠），用于在大小写不敏感的
        挂载上识别「同一个文件」。Windows 盘上的 ``Paper.pdf`` 与 ``paper.PDF``
        是同一个文件，只按原始路径比较会得出两个记录。
      * ``file_hash`` —— 内容哈希。文件被移动或改名时，靠它认出「还是同一篇」，
        而不是产生一条新记录 + 一条软删除记录。

    这三者合起来才是完整的身份判断，缺一个都会在 WSL 这种双端环境里出错。
    """

    __tablename__ = "papers"

    # --- 书目信息 ---
    title: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    title_norm: Mapped[str] = mapped_column(String(1024), nullable=False, default="", index=True)
    authors: Mapped[list | None] = mapped_column(JSON, default=list)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    venue: Mapped[str | None] = mapped_column(String(512), index=True)
    doi: Mapped[str | None] = mapped_column(String(255), index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), index=True)
    abstract: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(16))

    # --- 文件 ---
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    path_key: Mapped[str] = mapped_column(String(2048), nullable=False, index=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    file_size: Mapped[int | None] = mapped_column(Integer)
    file_mtime_ns: Mapped[int | None] = mapped_column(Integer)
    page_count: Mapped[int | None] = mapped_column(Integer)
    cover_path: Mapped[str | None] = mapped_column(String(1024))

    # --- 状态 ---
    source: Mapped[str] = mapped_column(String(16), nullable=False, default=SOURCE_LOCAL)
    ingest_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=INGEST_PENDING, index=True
    )
    reading_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=READING_UNREAD, index=True
    )
    rating: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)

    # --- 外部服务侧的文件句柄 ---
    # 上传到模型服务（如 Claude Files API）后拿到的 file_id。存在这里，
    # 是为了后续多轮问答 / 多阶段流水线不必反复重传 PDF 本体。
    # 带 provider 前缀是因为换模型供应商后旧 id 失效，需要重新上传。
    provider_file_id: Mapped[str | None] = mapped_column(String(255))
    provider_file_hash: Mapped[str | None] = mapped_column(
        String(64), comment="上传时的文件哈希；与 file_hash 不一致说明需要重新上传"
    )

    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    # --- 生命周期 ---
    # 软删除：文件从磁盘消失时只标记，不物理删除。笔记、标签、图关系都挂在
    # 论文上，级联删除会连带毁掉用户自己写的内容——那是不可接受的。
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    # --- 关系 ---
    # 注意 notes 的级联：论文被删时**不能**连带删掉笔记。笔记里有人工撰写的内容，
    # 那是最贵的数据。所以这里只做 save-update，删除时由业务层决定笔记的去留。
    notes: Mapped[list[Note]] = relationship(
        back_populates="paper", cascade="save-update, merge", lazy="selectin"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", passive_deletes=True
    )
    repos: Mapped[list[CodeRepo]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", passive_deletes=True
    )
    tag_links: Mapped[list[PaperTag]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        # 内容哈希不唯一：同一篇论文在不同目录下各有一份是常见情况，
        # 我们要能查出来（用于去重提示），但不能阻止入库。
        Index("ix_papers_hash_alive", "file_hash", "deleted_at"),
        Index("ix_papers_reading_added", "reading_status", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<Paper {self.id} {self.title[:40]!r}>"

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    @property
    def has_file(self) -> bool:
        import os

        return bool(self.file_path) and os.path.isfile(self.file_path)


class PaperDuplicate(IdMixin, TimestampMixin, Base):
    """疑似重复的论文对。

    自动去重只处理「确定是同一篇」的情况（内容哈希相同、DOI 相同）。
    标题相似度这类模糊判断一律落到这张表里等人确认——误合并会静默丢掉一篇
    真实存在的论文，代价远高于让用户点几下。
    """

    __tablename__ = "paper_duplicates"

    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )

    reason: Mapped[str] = mapped_column(String(32), nullable=False)  # exact_hash/doi/title_fuzzy
    score: Mapped[float] = mapped_column(Float, default=1.0)
    detail: Mapped[dict | None] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True
    )  # pending / merged / dismissed
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_by: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint("paper_id", "candidate_id", "reason", name="uq_dup_pair_reason"),
    )


class ScanRun(IdMixin, TimestampMixin, Base):
    """一次目录扫描的记录。用于回答「上次扫了什么、扫出什么」。"""

    __tablename__ = "scan_runs"

    root: Mapped[str] = mapped_column(String(2048), nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(26), index=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    files_seen: Mapped[int] = mapped_column(Integer, default=0)
    files_added: Mapped[int] = mapped_column(Integer, default=0)
    files_moved: Mapped[int] = mapped_column(Integer, default=0)
    files_missing: Mapped[int] = mapped_column(Integer, default=0)
    files_skipped: Mapped[int] = mapped_column(Integer, default=0)
    duplicates_found: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(16), default="running")
    error: Mapped[str | None] = mapped_column(Text)
    stats: Mapped[dict | None] = mapped_column(JSON, default=dict)


class CodeRepo(IdMixin, TimestampMixin, Base):
    """论文对应的开源代码仓库。

    本期只做「关联 + 浅索引」：识别出仓库、记录来源与匹配依据，
    真正把论文概念映射到代码文件（复现要点/超参对照）是后续阶段的事。
    """

    __tablename__ = "code_repos"

    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )

    name: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    url: Mapped[str | None] = mapped_column(String(1024))
    local_path: Mapped[str | None] = mapped_column(String(2048), index=True)
    vcs: Mapped[str | None] = mapped_column(String(16))  # git / hg / none
    head_commit: Mapped[str | None] = mapped_column(String(64))
    default_branch: Mapped[str | None] = mapped_column(String(128))

    language_stats: Mapped[dict | None] = mapped_column(JSON, default=dict)
    readme_excerpt: Mapped[str | None] = mapped_column(Text)

    # 关联是怎么建立的：正文里发现的链接、本地目录名匹配、还是人工指定。
    # 保留这个是因为「为什么这两者被认为相关」会直接影响用户对笔记的信任度。
    mapping_source: Mapped[str | None] = mapped_column(String(32))  # paper_text/dir_match/manual
    mapping_confidence: Mapped[float | None] = mapped_column(Float)
    mapping_evidence: Mapped[dict | None] = mapped_column(JSON, default=dict)

    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    paper: Mapped[Paper | None] = relationship(back_populates="repos")

    __table_args__ = (
        # 同一个仓库可以关联到多篇论文（一系列工作共用一个代码库），
        # 但同一对 (论文, 本地路径) 不该重复。
        UniqueConstraint("paper_id", "local_path", name="uq_repo_paper_path"),
    )


__all__ = [
    "INGEST_STATUSES",
    "READING_STATUSES",
    "SOURCES",
    "CodeRepo",
    "Paper",
    "PaperDuplicate",
    "ScanRun",
]
