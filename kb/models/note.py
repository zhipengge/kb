"""笔记模型。

笔记是本系统里**最有价值也最脆弱**的数据：论文可以重新下载，索引可以重建，
但人写的理解丢了就没了。这个判断贯穿了整个设计：

  * 论文记录被删除时不级联删除笔记（见 paper.py 的 relationship 注释）；
  * 笔记正文双写——数据库供检索，Markdown 文件供用户在 Obsidian / 编辑器里直接读写；
  * 冲突时**不自动覆盖任何一侧**，而是生成冲突副本让人决定；
  * 每次写入前留版本快照，可回滚。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:
    from .paper import Paper
    from .tag import NoteTag

# 笔记类型
KIND_SUMMARY = "summary"          # 摘要式概览
KIND_DEEP_READ = "deep_read"      # 深度阅读产出
KIND_CODE_REVIEW = "code_review"  # 论文 ↔ 代码对照
KIND_QA = "qa"                    # 问答沉淀
KIND_INSIGHT = "insight"          # 自己的想法
KIND_MANUAL = "manual"            # 手写
KINDS = (KIND_SUMMARY, KIND_DEEP_READ, KIND_CODE_REVIEW, KIND_QA, KIND_INSIGHT, KIND_MANUAL)

SOURCE_AI = "ai"
SOURCE_HUMAN = "human"
SOURCE_HYBRID = "hybrid"
SOURCE_IMPORT = "import"
SOURCES = (SOURCE_AI, SOURCE_HUMAN, SOURCE_HYBRID, SOURCE_IMPORT)

STATUS_DRAFT = "draft"
STATUS_PUBLISHED = "published"
STATUSES = (STATUS_DRAFT, STATUS_PUBLISHED)


class Note(IdMixin, TimestampMixin, Base):
    """一篇笔记。

    身份与位置分离：``id`` 是稳定的，``file_path`` 会变。用户在 Obsidian 里
    重命名文件、把笔记挪到别的目录，都不应该让笔记变成两条记录，也不该丢掉
    它的标签和双链。做到这一点的关键是写进 frontmatter 的 ``kb_id``——
    文件内容里带着自己的身份，比任何基于路径或 mtime 的推断都可靠。
    """

    __tablename__ = "notes"

    # 可以为空：允许存在与具体论文无关的独立笔记（读书笔记、技术总结）。
    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="SET NULL"), index=True
    )

    kind: Mapped[str] = mapped_column(String(32), nullable=False, default=KIND_MANUAL, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    slug: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    content_md: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # --- 落盘同步 ---
    file_path: Mapped[str | None] = mapped_column(String(2048), index=True)
    file_hash: Mapped[str | None] = mapped_column(String(64))
    # 我们自己写入文件后记下的 mtime。下次比对时若磁盘 mtime 与此不同，
    # 说明文件被外部改动过。没有这个「预期值」，就无法区分
    # 「文件是我们刚写的」和「文件被别人改了」。
    written_mtime_ns: Mapped[int | None] = mapped_column(Integer)
    sync_state: Mapped[str] = mapped_column(
        String(16), nullable=False, default="synced", index=True
    )  # synced / dirty / conflict / missing / external
    sync_error: Mapped[str | None] = mapped_column(Text)

    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=STATUS_DRAFT, index=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default=SOURCE_HUMAN)

    # 生成溯源：哪一版提示词、哪个模型产出的。出问题时要能定位到是模型换了
    # 还是提示词改了；也让「重跑一遍」有据可依。
    model: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    paper: Mapped[Paper | None] = relationship(back_populates="notes")
    revisions: Mapped[list[NoteRevision]] = relationship(
        back_populates="note",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="NoteRevision.version.desc()",
    )
    tag_links: Mapped[list[NoteTag]] = relationship(
        back_populates="note", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        # 同一篇论文下 slug 唯一，保证落盘文件名不打架
        UniqueConstraint("paper_id", "slug", name="uq_note_paper_slug"),
        Index("ix_notes_updated", "updated_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Note {self.id} {self.title[:40]!r}>"

    @property
    def is_ai_generated(self) -> bool:
        return self.source in (SOURCE_AI, SOURCE_HYBRID)


class NoteRevision(IdMixin, TimestampMixin, Base):
    """笔记的历史版本。

    每次内容变更前留一份快照。AI 重写笔记是常见操作，而模型每次产出都不一样；
    没有版本历史的话，用户点了「重新生成」就等于永久放弃上一版——
    这种不可逆操作会让人不敢用这个功能。
    """

    __tablename__ = "note_revisions"

    note_id: Mapped[str] = mapped_column(
        ForeignKey("notes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    content_md: Mapped[str] = mapped_column(Text, nullable=False, default="")

    author: Mapped[str | None] = mapped_column(String(64))  # human / ai / system
    summary: Mapped[str | None] = mapped_column(String(512))  # 变更摘要，用于 diff 列表
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    note: Mapped[Note] = relationship(back_populates="revisions")

    __table_args__ = (UniqueConstraint("note_id", "version", name="uq_revision_note_version"),)


__all__ = [
    "KINDS",
    "KIND_DEEP_READ",
    "KIND_MANUAL",
    "SOURCES",
    "STATUSES",
    "Note",
    "NoteRevision",
]
