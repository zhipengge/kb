"""AI 会话模型。

消息内容按**内容块列表**存储（而不是一个字符串），因为一轮回答里可能同时有：
思考摘要、正文、工具调用、引用块、图片。用字符串存的话，这些信息在落库那一刻
就丢了，前端只能显示一段裸文本——而「引用了哪篇论文的哪一页」正是本系统的核心价值。

存储格式与 Anthropic 的 content block 结构保持一致，这样历史消息可以
原样回传给模型，不需要在每次请求时做一次有损转换（尤其是 thinking 块和
tool_use/tool_result 配对，转换时极易破坏）。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, IdMixin, TimestampMixin

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_TOOL = "tool"
ROLES = (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_TOOL)


class Conversation(IdMixin, TimestampMixin, Base):
    """一次会话。"""

    __tablename__ = "conversations"

    title: Mapped[str] = mapped_column(String(512), nullable=False, default="新会话")
    model: Mapped[str | None] = mapped_column(String(128))
    system_prompt: Mapped[str | None] = mapped_column(Text)

    # 会话级别的限定范围：只在这几篇论文里检索。
    # 「围绕某个课题聊」是很自然的用法，没有这个字段就只能靠模型自己记住。
    scope_paper_ids: Mapped[list | None] = mapped_column(JSON, default=list)

    # 允许的工具名列表。存下来是为了回放历史会话时行为一致——
    # 如果工具集在两次访问之间变了，同样的提问会得到不同的答案。
    enabled_tools: Mapped[list | None] = mapped_column(JSON, default=list)

    pinned: Mapped[bool] = mapped_column(default=False)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Conversation {self.id} {self.title[:30]!r}>"


class Message(IdMixin, TimestampMixin, Base):
    """一条消息。

    ``citations`` 是结构化存储的引用，形如::

        [{"marker": 1, "paper_id": "01H...", "chunk_id": "01H...",
          "page": 3, "section": "3 Method", "quote": "...",
          "verified": true}]

    回答正文里出现的是 ``[1]`` 这样的标记，标记到具体位置的映射由服务端持有。
    这样做而不是让模型直接写论文标题，是因为**模型写不对标识符**——
    而服务端生成的映射可以保证每个标记都真实指向库里存在的分块，
    并且能校验引文是否真的出现在被引分块中（见 services/rag.py 的落地校验）。
    """

    __tablename__ = "messages"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    role: Mapped[str] = mapped_column(String(16), nullable=False, default=ROLE_USER)

    # 内容块列表，格式与模型 API 的 content blocks 对齐
    content: Mapped[list | None] = mapped_column(JSON, default=list)
    # 纯文本副本，用于全文检索与列表预览（从 content 派生，便于查询）
    text: Mapped[str | None] = mapped_column(Text)

    citations: Mapped[list | None] = mapped_column(JSON, default=list)
    tool_calls: Mapped[list | None] = mapped_column(JSON, default=list)

    # token 用量与成本，用于「这次回答花了多少钱」
    usage: Mapped[dict | None] = mapped_column(JSON, default=dict)
    model: Mapped[str | None] = mapped_column(String(128))
    stop_reason: Mapped[str | None] = mapped_column(String(32))

    # 分支：从某条消息重新生成时，新消息的 parent 指向它，
    # 这样同一会话里可以保留多个回答版本而不互相覆盖
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), index=True
    )

    elapsed_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_messages_conv_created", "conversation_id", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Message {self.id} {self.role} {(self.text or '')[:30]!r}>"


__all__ = [
    "ROLES",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "Conversation",
    "Message",
]
