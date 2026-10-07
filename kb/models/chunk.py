"""检索域模型：文本分块与嵌入模型登记。

``Chunk`` 是整个系统里最容易被低估的一张表。它不只是「分块后的文本」，
而是**引用定位的载体**：聊天回答里的「[论文 X 第 3 页 §2.1]」能成立，
完全依赖于每个 chunk 都记着自己来自哪一篇的哪一页哪一节。

所以 chunk 的切分方式不是「每 800 字切一刀」，而是沿章节边界切，
并且把页码与章节路径作为一等字段存下来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
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

# 分块类型。分开是因为不同类别的检索价值差别很大——
# 图表标题对「这个指标是多少」有用，对「方法是什么」没用。
CHUNK_TEXT = "text"
CHUNK_TABLE = "table"
CHUNK_FIGURE = "figure"      # 图注
CHUNK_FORMULA = "formula"
CHUNK_CODE = "code"
CHUNK_REFERENCE = "reference"
CHUNK_ABSTRACT = "abstract"
# 笔记正文按小节切出来的块。
#
# **笔记必须进索引，否则中文提问只能靠模型翻成英文才够得着语料。**
# 论文正文是英文的，中文问句在全文检索里一条都匹配不上，唯一的通路是
# 查询扩展那一步 LLM 调用。于是「模型不可用」（超预算、断网、没配 key）
# 等于「中文问句全部搜不到」，而界面上只显示「没有找到相关内容」。
# 笔记是中文写的，把它们索引进来，这条通路就不再依赖任何模型调用。
CHUNK_NOTE = "note"
CHUNK_KINDS = (
    CHUNK_ABSTRACT,
    CHUNK_CODE,
    CHUNK_FIGURE,
    CHUNK_FORMULA,
    CHUNK_NOTE,
    CHUNK_REFERENCE,
    CHUNK_TABLE,
    CHUNK_TEXT,
)


class Chunk(IdMixin, TimestampMixin, Base):
    """一段可检索、可引用的文本。"""

    __tablename__ = "chunks"

    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    note_id: Mapped[str | None] = mapped_column(
        ForeignKey("notes.id", ondelete="CASCADE"), index=True
    )

    kind: Mapped[str] = mapped_column(String(16), nullable=False, default=CHUNK_TEXT, index=True)

    # --- 定位信息（引用能力的基石）---
    # section_path 形如 "3 Method > 3.2 Attention"，供展示与按节过滤
    section_path: Mapped[str | None] = mapped_column(String(512), index=True)
    section_index: Mapped[int | None] = mapped_column(Integer)
    page_from: Mapped[int | None] = mapped_column(Integer, index=True)  # 1-based，与 PDF 一致
    page_to: Mapped[int | None] = mapped_column(Integer)

    ord: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 文档内顺序

    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    n_tokens: Mapped[int | None] = mapped_column(Integer)

    # 该块是否是所在章节的开头——生成笔记时用来保证每节都有覆盖
    is_section_start: Mapped[bool] = mapped_column(Boolean, default=False)

    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    paper: Mapped[Paper | None] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_paper_ord", "paper_id", "ord"),
        Index("ix_chunks_note_ord", "note_id", "ord"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Chunk {self.id} p{self.page_from} {self.text[:30]!r}>"

    @property
    def locator(self) -> str:
        """人类可读的定位串，直接进引用标记。"""
        bits = []
        if self.section_path:
            bits.append(f"§{self.section_path}")
        if self.page_from:
            bits.append(f"p.{self.page_from}" if not self.page_to or self.page_to == self.page_from
                        else f"p.{self.page_from}-{self.page_to}")
        return " ".join(bits)


class EmbeddingModel(IdMixin, TimestampMixin, Base):
    """一个已登记使用的嵌入模型，以及它在库里对应的向量表。

    换嵌入模型是必然发生的事（换供应商、换更小的模型、模型升级），而不同模型的
    向量维度不同、语义空间也不通用，**不能混在同一个向量表里**。

    做法是：每个 (provider, model, dim) 组合对应一张独立的 vec 表。
    换模型时新建一张表并重新嵌入，旧表原地保留——这样既能随时回退，
    也能对比两个模型的检索效果，而不是被迫做一次不可逆的替换。
    """

    __tablename__ = "embedding_models"

    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)

    table_name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # 该模型实际嵌入过的分块数，用于进度显示与「是否已完全重嵌入」的判断
    embedded_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("provider", "model", "dim", name="uq_embedding_identity"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<EmbeddingModel {self.slug} dim={self.dim}>"


__all__ = [
    "CHUNK_KINDS",
    "CHUNK_NOTE",
    "CHUNK_TEXT",
    "Chunk",
    "EmbeddingModel",
]
