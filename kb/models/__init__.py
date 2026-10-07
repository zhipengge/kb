"""全部模型的集中导入点。

模型的 relationship 用字符串声明，由 SQLAlchemy 的类注册表在 mapper 配置期
解析。因此**必须**有这么一个地方把所有模块都导入一遍，否则某个模型没被加载时，
指向它的 relationship 会在首次使用时才报错——那时候报错的地方离原因很远。

导入顺序不敏感：各模块之间只用字符串互相引用，没有运行期循环依赖。
"""

from .base import Base, IdMixin, TimestampMixin, new_id, utcnow
from .chat import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLES,
    Conversation,
    Message,
)
from .chunk import (
    CHUNK_KINDS,
    CHUNK_TEXT,
    Chunk,
    EmbeddingModel,
)
from .graph import (
    ENT_METHOD,
    ENTITY_TYPES,
    RELATION_TYPES,
    Entity,
    PaperEntity,
    Relation,
)
from .note import (
    KIND_DEEP_READ,
    KIND_MANUAL,
    KINDS,
    Note,
    NoteFlagState,
    NoteRevision,
)
from .note import (
    SOURCES as NOTE_SOURCES,
)
from .note import (
    STATUSES as NOTE_STATUSES,
)
from .paper import (
    INGEST_STATUSES,
    READING_STATUSES,
    SOURCES,
    CodeRepo,
    Paper,
    PaperDuplicate,
    ScanRun,
)
from .system import (
    JOB_STATUSES,
    SCOPE_ADMIN,
    SCOPE_INGEST,
    SCOPE_READ,
    SCOPE_WRITE,
    SCOPES,
    TERMINAL_STATUSES,
    ApiKey,
    Artifact,
    AuditLog,
    Job,
    JobEvent,
    LLMUsage,
    Setting,
    WebSearchCache,
)
from .tag import (
    DIM_METHOD,
    DIM_MISC,
    DIM_STATUS,
    DIM_TASK,
    DIM_TOPIC,
    DIMENSIONS,
    NoteTag,
    PaperTag,
    Tag,
    TagSuggestion,
)

__all__ = [
    "CHUNK_KINDS",
    "CHUNK_TEXT",
    "DIMENSIONS",
    "DIM_METHOD",
    "DIM_MISC",
    "DIM_STATUS",
    "DIM_TASK",
    "DIM_TOPIC",
    "ENTITY_TYPES",
    "ENT_METHOD",
    "INGEST_STATUSES",
    "JOB_STATUSES",
    "KINDS",
    "KIND_DEEP_READ",
    "KIND_MANUAL",
    "NOTE_SOURCES",
    "NOTE_STATUSES",
    "READING_STATUSES",
    "RELATION_TYPES",
    "ROLES",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "SCOPES",
    "SCOPE_ADMIN",
    "SCOPE_INGEST",
    "SCOPE_READ",
    "SCOPE_WRITE",
    "SOURCES",
    "TERMINAL_STATUSES",
    "ApiKey",
    "Artifact",
    "AuditLog",
    "Base",
    "Chunk",
    "CodeRepo",
    "Conversation",
    "EmbeddingModel",
    "Entity",
    "IdMixin",
    "Job",
    "JobEvent",
    "LLMUsage",
    "Message",
    "Note",
    "NoteFlagState",
    "NoteRevision",
    "NoteTag",
    "Paper",
    "PaperDuplicate",
    "PaperEntity",
    "PaperTag",
    "Relation",
    "ScanRun",
    "Setting",
    "Tag",
    "TagSuggestion",
    "TimestampMixin",
    "WebSearchCache",
    "new_id",
    "utcnow",
]
