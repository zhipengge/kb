"""向量嵌入。

设计前提是：**没配置嵌入模型时，整个系统必须完全可用**，只是检索退化为
纯全文检索。这是默认状态——用户装好就能搜，不必先去申请一个 API Key。
所以这里的每个入口都先检查「配置了吗」，而不是假设一定有。

向量存储有两种形态，按环境能力选择，由 ``embedding_models.meta["storage"]`` 记录：

  * ``vec0``  —— sqlite-vec 扩展可用时。虚拟表自带 KNN 索引，查询快。
  * ``blob``  —— 扩展不可用时。向量以 float32 字节串存在普通表里，
    检索时用 numpy 全量算余弦相似度。万级分块下几十毫秒，可以接受。

两种形态对上层暴露同一个查询接口，调用方不需要知道底层是哪一种。
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
import time
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import text

from ..extensions import db
from ..models import Chunk, EmbeddingModel
from ..utils.sql import safe_identifier
from .paths import slugify

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 提供方
# --------------------------------------------------------------------------


class EmbeddingError(RuntimeError):
    """嵌入计算失败。消息面向用户。"""


class EmbeddingProvider(Protocol):
    slug: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class OpenAICompatEmbedder:
    """OpenAI 兼容的嵌入接口。

    DeepSeek、Qwen、SiliconFlow、vLLM、Ollama、LM Studio 等
    都提供这一形状的接口，所以一个实现能覆盖绝大多数自建/第三方服务。
    """

    model: str
    api_key: str
    base_url: str | None = None
    dim: int = 0
    batch_size: int = 32
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.slug = slugify(f"{self.model}-{self.dim or 'auto'}", fallback="embed")
        if not self.model:
            raise EmbeddingError("没有配置嵌入模型")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url or None,
            timeout=self.timeout,
        )

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            # 空字符串会让部分服务端报错，统一替换成一个空格
            batch = [t if t and t.strip() else " " for t in batch]
            try:
                response = client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:
                raise EmbeddingError(f"调用嵌入接口失败：{exc}") from exc

            # 服务端不保证按顺序返回，必须按 index 排序后再取
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)

            if not self.dim and ordered:
                self.dim = len(ordered[0].embedding)

        return vectors


class LocalEmbedder:
    """本地 ONNX 嵌入（fastembed）。

    选它而不是 ``sentence-transformers``：后者会拖进约 2GB 的 torch，
    对一个「磁盘上的个人知识库」来说太重。fastembed 走 ONNX Runtime，
    体积小得多，且模型下载后可以完全离线跑。

    默认模型 ``bge-small-zh-v1.5`` 是中文模型，但它对英文的编码也够用——
    实测「变分自编码器的优化目标是什么」能正确匹配到英文的
    "evidence lower bound" 那段。这是本项目里中文提问命中英文论文的
    主要手段：纯全文检索做不到跨语言，只能靠其中的技术词碰运气。
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-zh-v1.5",
        dim: int = 0,
        batch_size: int = 16,
    ):
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.slug = slugify(f"local-{model}-{dim or 'auto'}", fallback="local-embed")
        # 模型实例缓存。**必须缓存**：TextEmbedding(...) 会加载 ONNX 模型文件，
        # 每次调用都新建的话，批量嵌入 178 个分块要重复加载十几次模型，
        # 开销远超推理本身。
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingError(
                "本地嵌入需要 fastembed。安装：pipenv install fastembed\n"
                "或者改用 OpenAI 兼容的远程嵌入接口（设置 → 向量嵌入）。"
            ) from exc

        try:
            self._encoder = TextEmbedding(model_name=self.model)
        except Exception as exc:
            raise EmbeddingError(
                f"加载本地嵌入模型 {self.model} 失败：{exc}\n"
                "首次使用需要联网下载模型文件；如果网络受限，"
                "可以设 HF_ENDPOINT=https://hf-mirror.com 使用镜像。"
            ) from exc
        return self._encoder

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoder = self._get_encoder()
        vectors = [list(map(float, vec)) for vec in encoder.embed(texts)]
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors


def get_provider() -> EmbeddingProvider | None:
    """按当前设置构造嵌入提供方。未配置则返回 None。"""
    from flask import current_app

    settings = current_app.extensions["kb_settings"]
    provider = settings.get("embedding.provider")
    model = (settings.get("embedding.model") or "").strip()
    if not model:
        return None

    if provider == "local":
        return LocalEmbedder(
            model=model,
            dim=int(settings.get("embedding.dim") or 0),
            batch_size=int(settings.get("embedding.batch_size")),
        )


    api_key = settings.get_secret("embedding.api_key")
    base_url = (settings.get("embedding.base_url") or "").strip()
    if not api_key and not base_url:
        # 本地推理服务（Ollama / vLLM）通常不校验 key，只要给了地址就能用
        return None

    return OpenAICompatEmbedder(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        dim=int(settings.get("embedding.dim") or 0),
        batch_size=int(settings.get("embedding.batch_size")),
    )


def is_configured() -> bool:
    try:
        return get_provider() is not None
    except Exception:
        return False


# --------------------------------------------------------------------------
# 向量表管理
# --------------------------------------------------------------------------


def _vec_available() -> bool:
    from flask import current_app

    state = current_app.extensions.get("kb_vector_state") or {}
    return bool(state.get("enabled"))


def _table_name(slug: str, storage: str) -> str:
    """由模型 slug 生成向量表名。

    slug 来自模型名（``BAAI/bge-small-zh-v1.5`` 之类），里面有点、斜杠、
    连字符——这些在 SQL 标识符里都非法，直接拼进 CREATE TABLE 会得到
    一个难懂的语法错误。统一换成下划线。

    **截断之后必须补一段摘要。** 表名有唯一约束，而模型名常常共享长前缀：
    实测 ``…/paraphrase-multilingual-mpnet-base-v2`` 与
    ``…/paraphrase-multilingual-MiniLM-L12-v2`` 前 40 个字符完全相同，
    只做截断的话第二个模型**根本注册不进去**——报的是 UNIQUE 约束冲突，
    表现却是「这个模型换不过去」。留前缀是为了表名还能认出是哪个模型，
    加摘要才是保证不撞车的那一半。
    """
    prefix = "vec_chunks_" if storage == "vec0" else "blob_chunks_"
    safe_slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_").lower()
    digest = hashlib.sha256(safe_slug.encode()).hexdigest()[:8]
    return f"{prefix}{safe_slug[:40]}_{digest}"


def _safe_table(model: EmbeddingModel) -> str:
    """取出并校验向量表名。

    表名存在数据库里、且是从用户配置的模型名派生出来的，属于「拼进 SQL 的
    标识符」。虽然 slugify 已经滤掉了危险字符，这里再过一道显式校验——
    见 utils/sql.py 里对这道防线的说明。
    """
    return safe_identifier(model.table_name)


def ensure_model_row(provider: EmbeddingProvider) -> EmbeddingModel | None:
    """确保 ``embedding_models`` 里有这个模型的登记，并建好它的向量表。"""
    storage = "vec0" if _vec_available() else "blob"
    slug = provider.slug

    row = db.session.query(EmbeddingModel).filter_by(slug=slug).one_or_none()
    if row is not None:
        return row

    dim = provider.dim
    if not dim:
        # 维度未知：先探测一次，否则建不出表。用一句短文本，成本可忽略。
        try:
            probe = provider.embed(["dimension probe"])
        except EmbeddingError:
            raise
        if not probe:
            raise EmbeddingError("无法确定嵌入维度")
        dim = len(probe[0])
        provider.dim = dim

    table_name = safe_identifier(_table_name(slug, storage))

    try:
        with db.engine.begin() as conn:
            if storage == "vec0":
                conn.execute(
                    text(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name} "
                        f"USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{dim}])"
                    )
                )
            else:
                conn.execute(
                    text(
                        f"CREATE TABLE IF NOT EXISTS {table_name} ("
                        f"chunk_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
                    )
                )
    except Exception as exc:
        raise EmbeddingError(f"创建向量表失败：{exc}") from exc

    # 新模型设为活跃，旧的降级为备选——保留旧表是为了能随时回退，
    # 而不是被迫做一次不可逆的替换
    db.session.query(EmbeddingModel).filter(EmbeddingModel.is_active.is_(True)).update(
        {"is_active": False}, synchronize_session=False
    )

    row = EmbeddingModel(
        slug=slug,
        provider="local" if isinstance(provider, LocalEmbedder) else "openai_compatible",
        model=provider.model,
        dim=dim,
        table_name=table_name,
        is_active=True,
        meta={"storage": storage},
    )
    db.session.add(row)
    db.session.commit()
    log.info("已登记嵌入模型 %s（%d 维，%s 存储）", slug, dim, storage)
    return row


def active_model() -> EmbeddingModel | None:
    return (
        db.session.query(EmbeddingModel)
        .filter(EmbeddingModel.is_active.is_(True))
        .order_by(EmbeddingModel.created_at.desc())
        .first()
    )


def _pack(vector: list[float]) -> bytes:
    """float32 紧凑字节串。用 float32 而不是 float64 是为了省一半空间——
    嵌入检索对精度不敏感，存储量却翻倍。"""
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def store_vectors(model: EmbeddingModel, pairs: list[tuple[str, list[float]]]) -> int:
    """写入向量。``pairs`` 是 ``[(chunk_id, vector)]``。"""
    if not pairs:
        return 0

    storage = (model.meta or {}).get("storage", "vec0")
    table = _safe_table(model)

    with db.engine.begin() as conn:
        if storage == "vec0":
            # vec0 的删除是按主键的，重复嵌入时先清掉旧值
            ids = [chunk_id for chunk_id, _ in pairs]
            for start in range(0, len(ids), 200):
                batch = ids[start : start + 200]
                placeholders = ",".join(f":id{i}" for i in range(len(batch)))
                conn.execute(
                    text(f"DELETE FROM {table} WHERE chunk_id IN ({placeholders})"),
                    {f"id{i}": value for i, value in enumerate(batch)},
                )
            for chunk_id, vector in pairs:
                conn.execute(
                    text(f"INSERT INTO {table}(chunk_id, embedding) VALUES (:cid, :vec)"),
                    {"cid": chunk_id, "vec": _serialize_for_vec(vector)},
                )
        else:
            for chunk_id, vector in pairs:
                conn.execute(
                    text(
                        f"INSERT INTO {table}(chunk_id, dim, vector) VALUES (:cid, :dim, :vec) "
                        f"ON CONFLICT(chunk_id) DO UPDATE SET dim=:dim, vector=:vec"
                    ),
                    {"cid": chunk_id, "dim": len(vector), "vec": _pack(vector)},
                )

    db.session.query(EmbeddingModel).filter_by(id=model.id).update(
        {"embedded_count": EmbeddingModel.embedded_count + len(pairs)}, synchronize_session=False
    )
    db.session.commit()
    return len(pairs)


def _serialize_for_vec(vector: list[float]) -> bytes:
    """sqlite-vec 接受 float32 的紧凑字节串。"""
    return struct.pack(f"<{len(vector)}f", *vector)


def delete_vectors(model: EmbeddingModel, chunk_ids: list[str]) -> int:
    if not chunk_ids:
        return 0
    table = _safe_table(model)
    removed = 0
    with db.engine.begin() as conn:
        for start in range(0, len(chunk_ids), 200):
            batch = chunk_ids[start : start + 200]
            placeholders = ",".join(f":id{i}" for i in range(len(batch)))
            result = conn.execute(
                text(f"DELETE FROM {table} WHERE chunk_id IN ({placeholders})"),
                {f"id{i}": value for i, value in enumerate(batch)},
            )
            removed += result.rowcount or 0
    return removed


def search_vectors(model: EmbeddingModel, query_vector: list[float], limit: int) -> list[tuple[str, float]]:
    """向量检索。返回 ``[(chunk_id, 相似度)]``，相似度越大越相关。

    vec0 用 L2 距离（越小越近），这里转换成相似度，与 blob 路径的余弦
    保持同一语义——上层不需要知道用的是哪种距离度量。
    """
    storage = (model.meta or {}).get("storage", "vec0")
    table = _safe_table(model)

    if storage == "vec0":
        with db.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT chunk_id, distance FROM {table} "
                    f"WHERE embedding MATCH :vec AND k = :k ORDER BY distance"
                ),
                {"vec": _serialize_for_vec(query_vector), "k": limit},
            ).fetchall()
        # L2 距离 -> 相似度：距离 0 映射到 1，越大越小
        return [(row[0], 1.0 / (1.0 + float(row[1]))) for row in rows]

    return _blob_search(table, model.dim, query_vector, limit)


def _blob_search(
    table: str, dim: int, query_vector: list[float], limit: int
) -> list[tuple[str, float]]:
    """numpy 全量余弦检索（sqlite-vec 不可用时的退路）。"""
    import numpy as np

    with db.engine.connect() as conn:
        rows = conn.execute(text(f"SELECT chunk_id, vector FROM {table}")).fetchall()

    if not rows:
        return []

    ids = [row[0] for row in rows]
    matrix = np.frombuffer(b"".join(row[1] for row in rows), dtype=np.float32).reshape(len(rows), dim)
    query = np.asarray(query_vector, dtype=np.float32)

    # 归一化后点积即余弦相似度
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        return []
    denominator = matrix_norms * query_norm
    denominator[denominator == 0] = 1.0
    scores = (matrix @ query) / denominator

    top = np.argsort(-scores)[:limit]
    return [(ids[i], float(scores[i])) for i in top]


# --------------------------------------------------------------------------
# 批量嵌入
# --------------------------------------------------------------------------


# 嵌入模型能吃的字符上限。
#
# 这个值不是随便定的：bge-small 系列的上限是 512 token，而我们的分块目标
# 是 800 token——**超出部分会被模型静默截断**，于是向量只代表块的开头，
# 块后半段的内容就检索不到了。这是最隐蔽的那类问题：不报错，
# 只是「有些内容死活搜不出来」。
EMBED_TEXT_LIMIT = 1200


def _embed_text(title: str, section: str, text: str) -> str:
    """构造实际送去嵌入的文本。

    **加上论文标题与章节路径前缀**，而不是只嵌入正文。两个理由：

    1. 分块脱离上下文后，一段讲「the strategy used in VAEs」的文字看不出
       指的是哪个策略。带上「论文标题 + 章节」之后语义完整得多。
    2. 用户检索时记得的往往是论文名或章节名，前缀让这类查询也能命中。

    嵌入的文本与展示的文本不必相同——这是检索里的常规做法。
    """
    parts = [p for p in (title, section) if p]
    prefix = " | ".join(parts)
    body = text[:EMBED_TEXT_LIMIT]
    return f"{prefix}\n{body}" if prefix else body


def embed_pending(ctx=None, paper_ids: list[str] | None = None, force: bool = False) -> dict:
    """为还没嵌入的分块计算向量。后台任务入口。"""
    provider = get_provider()
    if provider is None:
        return {
            "skipped": True,
            "reason": "未配置嵌入模型。设置 → 向量嵌入 里填写后可启用向量检索。",
        }

    model = ensure_model_row(provider)

    # 找出还没进向量表的分块
    query = db.session.query(Chunk)
    if paper_ids:
        query = query.filter(Chunk.paper_id.in_(paper_ids))
    chunks = query.order_by(Chunk.paper_id, Chunk.ord).all()

    existing: set[str] = set()
    if not force:
        try:
            with db.engine.connect() as conn:
                existing = {
                    row[0]
                    for row in conn.execute(text(f"SELECT chunk_id FROM {_safe_table(model)}"))
                }
        except Exception:
            existing = set()

    pending = [chunk for chunk in chunks if force or chunk.id not in existing]
    if not pending:
        return {"total": len(chunks), "embedded": 0, "skipped_existing": len(chunks)}

    batch_size = int(provider.batch_size) if hasattr(provider, "batch_size") else 32
    started = time.perf_counter()
    embedded = 0

    for start in range(0, len(pending), batch_size):
        if ctx is not None:
            ctx.check_cancelled()
            ctx.progress(
                start / len(pending),
                f"嵌入 {start}/{len(pending)}",
            )

        batch = pending[start : start + batch_size]

        # 每块带上它所属论文与章节，见 _embed_text 的说明
        payloads = [
            _embed_text(
                chunk.paper.title if chunk.paper else "",
                chunk.section_path or "",
                chunk.text,
            )
            for chunk in batch
        ]
        vectors = provider.embed(payloads)
        store_vectors(model, [(chunk.id, vec) for chunk, vec in zip(batch, vectors, strict=False)])
        embedded += len(batch)

    elapsed = time.perf_counter() - started
    log.info("嵌入完成：%d 个分块，用时 %.1fs", embedded, elapsed)
    return {
        "total": len(chunks),
        "embedded": embedded,
        "model": model.slug,
        "dim": model.dim,
        "elapsed_ms": int(elapsed * 1000),
    }


def embedding_stats() -> dict:
    """向量索引进度，供界面展示。"""
    model = active_model()
    if model is None:
        return {"configured": is_configured(), "model": None, "embedded": 0, "total": 0, "coverage": 0.0}

    total = db.session.query(Chunk).count()
    embedded = model.embedded_count or 0
    return {
        "configured": True,
        "model": model.slug,
        "provider": model.provider,
        "dim": model.dim,
        "storage": (model.meta or {}).get("storage", "vec0"),
        "embedded": embedded,
        "total": total,
        "coverage": round(embedded / total, 3) if total else 0.0,
    }


__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "LocalEmbedder",
    "OpenAICompatEmbedder",
    "active_model",
    "delete_vectors",
    "embed_pending",
    "embedding_stats",
    "ensure_model_row",
    "get_provider",
    "is_configured",
    "search_vectors",
    "store_vectors",
]
