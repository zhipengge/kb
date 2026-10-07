"""知识图谱：实体、关系、论文-实体关联。

设计上有一条硬规则：**每条关系都必须带证据**。

知识图谱最常见的失败模式是「看起来很美但没法验证」——模型抽出一堆
「A 改进了 B」的边，用户点开却不知道该不该信。所以 ``Relation.evidence``
记录支撑这条边的原文片段（chunk id + 引文），界面上可以一键跳回原文核对。
没有证据的边只可能来自人工确认。

本期完成建表与抽取接口；图的可视化交互在后续迭代（见 docs/roadmap.md）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Float,
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

# 实体类型
ENT_METHOD = "method"        # 方法/模型/算法
ENT_TASK = "task"            # 任务
ENT_DATASET = "dataset"      # 数据集
ENT_METRIC = "metric"        # 指标
ENT_CONCEPT = "concept"      # 概念
ENT_AUTHOR = "author"
ENT_ORG = "org"
ENT_VENUE = "venue"
ENT_TOOL = "tool"            # 软件/框架
ENTITY_TYPES = (
    ENT_AUTHOR,
    ENT_CONCEPT,
    ENT_DATASET,
    ENT_METRIC,
    ENT_METHOD,
    ENT_ORG,
    ENT_TASK,
    ENT_TOOL,
    ENT_VENUE,
)

# 关系类型
REL_PROPOSES = "proposes"          # 论文提出方法（src=论文实体化的方法节点, dst=…）
REL_USES = "uses"                  # A 使用 B
REL_IMPROVES = "improves"          # A 改进 B
REL_EXTENDS = "extends"
REL_COMPARES = "compares_with"     # A 与 B 对比
REL_EVALUATES_ON = "evaluates_on"  # A 在 B 上评测
REL_CITES = "cites"
REL_AUTHORED_BY = "authored_by"
REL_PUBLISHED_IN = "published_in"
REL_PART_OF = "part_of"
REL_RELATED = "related_to"
RELATION_TYPES = (
    REL_AUTHORED_BY,
    REL_CITES,
    REL_COMPARES,
    REL_EXTENDS,
    REL_EVALUATES_ON,
    REL_IMPROVES,
    REL_PART_OF,
    REL_PROPOSES,
    REL_PUBLISHED_IN,
    REL_RELATED,
    REL_USES,
)


class Entity(IdMixin, TimestampMixin, Base):
    """图谱节点。

    实体消歧（把 "ResNet"、"ResNet-50"、"resnet50" 认成同一个节点）靠三件事：
    归一化名字精确匹配、别名表、以及可选的向量相似度。三者都命中不了时
    宁可新建一个节点——错误合并两个不同方法，比留下两个重复节点危害大得多。
    """

    __tablename__ = "entities"

    name: Mapped[str] = mapped_column(String(512), nullable=False)
    name_norm: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False, default=ENT_CONCEPT, index=True)

    description: Mapped[str | None] = mapped_column(Text)
    aliases: Mapped[list | None] = mapped_column(JSON, default=list)

    # 出现次数，用于图上的节点大小
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)

    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("type", "name_norm", name="uq_entity_type_name"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Entity {self.type}:{self.name}>"


class Relation(IdMixin, TimestampMixin, Base):
    """图谱的边。"""

    __tablename__ = "relations"

    src_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dst_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # 这条边是从哪篇论文得到的。同一条边可能被多篇论文支持——
    # 那时的处理是更新 weight 并追加 evidence，而不是插入重复边。
    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )

    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    confidence: Mapped[float | None] = mapped_column(Float)

    # 证据：[{chunk_id, quote, page}]。没有证据的边在界面上会标为「未验证」。
    evidence: Mapped[list | None] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="ai")  # ai/manual

    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    src: Mapped[Entity] = relationship(foreign_keys=[src_id], lazy="joined")
    dst: Mapped[Entity] = relationship(foreign_keys=[dst_id], lazy="joined")
    paper: Mapped[Paper | None] = relationship()

    __table_args__ = (
        # 同一对节点 + 同一关系类型只保留一条边（多篇论文支持时累加权重与证据）
        UniqueConstraint("src_id", "dst_id", "type", name="uq_relation_triple"),
        Index("ix_relations_type_weight", "type", "weight"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Relation {self.src_id[:6]}-{self.type}->{self.dst_id[:6]}>"

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


class PaperEntity(Base):
    """论文 ↔ 实体。单独一张表是为了「这篇论文涉及哪些方法/数据集」这类查询能走索引。

    关系表里也有 paper_id，但那只能回答「这条边从哪来」。
    要在论文页面上列出「本文提出的方法、使用的数据集、对比的基线」，
    需要的是论文到实体的直接映射。
    """

    __tablename__ = "paper_entities"

    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )

    # 这篇论文在该实体上扮演的角色：proposes / uses / evaluates_on / baseline
    role: Mapped[str | None] = mapped_column(String(32), index=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    paper: Mapped[Paper] = relationship()
    entity: Mapped[Entity] = relationship(lazy="joined")


__all__ = [
    "ENTITY_TYPES",
    "ENT_METHOD",
    "RELATION_TYPES",
    "REL_IMPROVES",
    "REL_USES",
    "Entity",
    "PaperEntity",
    "Relation",
]
