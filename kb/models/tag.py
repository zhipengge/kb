"""标签体系：受控词表 + 关联 + AI 建议队列。

为什么是「受控词表」而不是「随便打标」：让模型自由生成标签，几百篇论文之后
你会得到 3000 个只用过一次的标签（"attention mechanism"、"attention-mechanism"、
"Attention Mechanisms"、"注意力机制"…），标签就彻底失去了聚合能力。

所以这里的规则是：
  * 标签有 ``dimension``（维度）——同一个词在不同维度下是不同标签，
    比如 "Transformer" 作为 *方法* 和作为 *架构* 是两回事；
  * 标签有 ``aliases``（别名）——模型和用户写的各种变体都归一到同一个标签；
  * 标签有层级（``parent_id``）——支持「深度学习/注意力机制/自注意力」这样的树；
  * AI **不能直接创建标签**，只能写进 ``tag_suggestions`` 等人确认。
"""

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

if TYPE_CHECKING:
    from .note import Note
    from .paper import Paper

# 标签维度。分开是为了让筛选器有意义——
# 「2023 年的、用扩散模型的、做图像分割的」是三个不同维度的约束。
DIM_TOPIC = "topic"        # 研究主题
DIM_METHOD = "method"      # 方法
DIM_TASK = "task"          # 任务
DIM_DOMAIN = "domain"      # 应用领域
DIM_VENUE = "venue"        # 发表场所
DIM_STATUS = "status"      # 个人状态：待读/在跟/已实现…
DIM_MISC = "misc"
DIMENSIONS = (DIM_TOPIC, DIM_METHOD, DIM_TASK, DIM_DOMAIN, DIM_VENUE, DIM_STATUS, DIM_MISC)

SOURCE_MANUAL = "manual"
SOURCE_AI = "ai"
SOURCE_IMPORT = "import"


class Tag(IdMixin, TimestampMixin, Base):
    """受控词表中的一个标签。"""

    __tablename__ = "tags"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DIM_MISC, index=True
    )
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("tags.id", ondelete="SET NULL"), index=True
    )

    description: Mapped[str | None] = mapped_column(Text)
    color: Mapped[str | None] = mapped_column(String(16))

    # 别名用于归一：模型写 "self-attention"、用户写 "自注意力"，
    # 都指向同一个标签。检索时先查别名表再落到标签。
    aliases: Mapped[list | None] = mapped_column(JSON, default=list)

    # 冗余计数：标签云和筛选器要按热度排序，每次 COUNT(*) 太慢。
    # 由服务层在增删关联时维护，并提供 recompute 命令兜底。
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)

    is_auto: Mapped[bool] = mapped_column(default=False)  # 是否为 AI 建议创建的
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    parent: Mapped[Tag | None] = relationship(remote_side="Tag.id", back_populates="children")
    children: Mapped[list[Tag]] = relationship(back_populates="parent")

    paper_links: Mapped[list[PaperTag]] = relationship(
        back_populates="tag", cascade="all, delete-orphan", passive_deletes=True
    )
    note_links: Mapped[list[NoteTag]] = relationship(
        back_populates="tag", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        # 同维度下不允许重名或重 slug——这是「受控」的核心约束
        UniqueConstraint("dimension", "slug", name="uq_tag_dimension_slug"),
        UniqueConstraint("dimension", "name", name="uq_tag_dimension_name"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tag {self.dimension}:{self.name}>"

    @property
    def full_path(self) -> str:
        """形如 ``方法 / 注意力机制 / 自注意力``，用于筛选器与展示。"""
        parts = [self.name]
        node = self.parent
        seen = {self.id}
        while node is not None and node.id not in seen:  # seen 防御脏数据造成的环
            seen.add(node.id)
            parts.append(node.name)
            node = node.parent
        return " / ".join(reversed(parts))


class PaperTag(Base):
    """论文 ↔ 标签。带来源与置信度，因为「我确认的」和「模型猜的」不该等价。"""

    __tablename__ = "paper_tags"

    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[str] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)

    source: Mapped[str] = mapped_column(String(16), nullable=False, default=SOURCE_MANUAL)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    paper: Mapped[Paper] = relationship(back_populates="tag_links")
    tag: Mapped[Tag] = relationship(back_populates="paper_links", lazy="joined")


class NoteTag(Base):
    """笔记 ↔ 标签。"""

    __tablename__ = "note_tags"

    note_id: Mapped[str] = mapped_column(
        ForeignKey("notes.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[str] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)

    source: Mapped[str] = mapped_column(String(16), nullable=False, default=SOURCE_MANUAL)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    note: Mapped[Note] = relationship(back_populates="tag_links")
    tag: Mapped[Tag] = relationship(back_populates="note_links", lazy="joined")


class TagSuggestion(IdMixin, TimestampMixin, Base):
    """AI 建议的标签，等待人工裁决。

    这是「防止标签爆炸」的执行点。模型可以自由地提议，但提议要落到这里，
    由人决定是接受（并入词表/关联到论文）还是拒绝。拒绝记录同样保留——
    下次同类论文再提同一个词时可以直接过滤掉，不会反复打扰。
    """

    __tablename__ = "tag_suggestions"

    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    note_id: Mapped[str | None] = mapped_column(
        ForeignKey("notes.id", ondelete="CASCADE"), index=True
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[str] = mapped_column(String(32), nullable=False, default=DIM_MISC)

    # 如果建议的其实是已有标签的别名，记下它——UI 上就能展示成
    # 「建议：自注意力（已有标签 注意力机制 的别名）」，而不是让用户以为要新建。
    suggested_tag_id: Mapped[str | None] = mapped_column(
        ForeignKey("tags.id", ondelete="SET NULL")
    )
    rationale: Mapped[str | None] = mapped_column(Text)  # 模型的理由，简短
    evidence: Mapped[dict | None] = mapped_column(JSON, default=dict)

    confidence: Mapped[float | None] = mapped_column(Float)
    model: Mapped[str | None] = mapped_column(String(128))

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True
    )  # pending / accepted / rejected
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # 同一目标上同一个词只留一条待裁决建议，避免重复刷屏
        Index("ix_suggestion_target", "paper_id", "note_id", "name", "status"),
    )


__all__ = [
    "DIMENSIONS",
    "DIM_METHOD",
    "DIM_MISC",
    "DIM_STATUS",
    "DIM_TASK",
    "DIM_TOPIC",
    "SOURCE_AI",
    "SOURCE_MANUAL",
    "NoteTag",
    "PaperTag",
    "Tag",
    "TagSuggestion",
]
