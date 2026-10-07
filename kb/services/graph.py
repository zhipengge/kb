"""知识图谱的构建与查询。

`kb/models/graph.py` 里那套表结构早就设计好了（9 类实体、11 类关系，
连实体消歧策略都写清楚了），但一直**没有任何代码写过一行数据**——
`/graph` 路由是 404，Cytoscape 的 434KB 也躺在 vendor 里没接过。这里把它接上。

## 为什么每条关系都要带证据

图谱最容易变成「看起来很美但没法验证」：模型抽出一堆「A 改进了 B」的边，
用户点开不知道该不该信，几次之后就再也不看了。所以每条边都记下
**支撑它的原句**，界面上能直接看到——看得见出处的边才有资格参与判断。

## 证据取自**笔记**而不是论文原文

笔记是已经消化过的中文，抽出来的关系密度高得多；而且笔记本身有版本、有出处，
回查是通的。代价是：证据是「笔记这么说」，不是「论文这么说」。
这是**有意的取舍**，不是疏漏——界面上的措辞要如实反映这一点，
不能把它显示成「已核对原文」。

## 已知边界

抽取用的是通用提示词，不做领域微调。实测同一篇论文抽两次，实体名会有出入
（"BEV" vs "bird's eye view"），靠 `name_norm` + 别名表收敛；收敛不了就新建节点。
**宁可多一个重复节点，也不要把两个不同的方法合成一个**——后者会让人得出
「这两个方法是一回事」的错误结论，而前者只是看起来乱。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

PROMPT_VERSION = "2026-10-08.1"

SYSTEM = """你在为一个论文知识库构建知识图谱。

给定一篇论文的**精读笔记**（中文），抽出其中的实体与它们之间的关系。

**实体**：方法/模型、任务、数据集、指标、概念、工具。
  * 名字用**论文里的规范写法**（通常是英文），中文名放 aliases。
  * 只抽**具体、可指认**的东西。"模型"、"方法" 这种泛词不要抽。

**关系**：只抽笔记里**明确支持**的，不要靠常识补全。可用类型：
    proposes        提出的方法（src 用论文标题里的方法名）
    uses            使用了某方法/工具/数据集
    improves        改进了某个已有方法
    extends         扩展了某个方法
    compares_with   与某方法对比
    evaluates_on    在某数据集上评测
    part_of         属于某个更大的框架/任务

**每条关系必须带 evidence**：从笔记里**原样摘一句**支持它的话（照抄，不要改写）。
摘不出原句的关系就不要输出——宁可少一条边，也不要一条没有出处的边。

输出 JSON：
{"entities":[{"name":"...","type":"method","aliases":["..."]}],
 "relations":[{"src":"...","dst":"...","type":"uses","evidence":"原句"}]}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "method", "task", "dataset", "metric",
                            "concept", "tool",
                        ],
                    },
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "proposes", "uses", "improves", "extends",
                            "compares_with", "evaluates_on", "part_of",
                        ],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["src", "dst", "type"],
            },
        },
    },
    "required": ["entities", "relations"],
}

# 名字归一化：小写、去掉标点与多余空白。
# **连字符要保留**——"end-to-end" 和 "end to end" 不该是两个节点，但
# "bge-m3" 也不该被拆成两个词。统一把分隔符压成单个空格即可。
_NORM_STRIP = re.compile(r"[^\w\s\-]+", re.UNICODE)
_NORM_WS = re.compile(r"\s+")

# 这些名字太泛，抽出来只会让图变成一团毛线
_STOPWORDS = {
    "model", "models", "method", "methods", "approach", "framework",
    "方法", "模型", "框架", "论文", "实验", "结果", "baseline", "sota",
}


def normalize(name: str) -> str:
    """实体名归一化。用于消歧时的精确匹配。"""
    text = _NORM_STRIP.sub(" ", (name or "").strip().lower())
    return _NORM_WS.sub(" ", text).strip()


def is_useful(name: str) -> bool:
    """过滤掉没有检索价值的泛词。"""
    norm = normalize(name)
    if len(norm) < 3:
        return False
    return norm not in _STOPWORDS


@dataclass
class Extraction:
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    error: str | None = None


def extract(note, provider=None) -> Extraction:
    """从一篇笔记里抽实体与关系。"""
    body = (note.content_md or "").strip()
    if len(body) < 200:
        return Extraction(error="笔记太短，跳过")

    if provider is None:
        from .llm import chat_provider

        provider = chat_provider()

    try:
        payload = provider.extract(
            [{"role": "user", "content": body[:12000]}],
            schema=SCHEMA,
            description="提交从笔记里抽出的实体与关系",
            system=SYSTEM,
            instructions="只输出 JSON。关系必须带 evidence（照抄笔记原句）。",
            max_tokens=8000,
        )
    except Exception as exc:
        log.warning("图谱抽取失败 %s：%s", note.id, exc)
        return Extraction(error=str(exc)[:200])

    return Extraction(
        entities=payload.get("entities") or [],
        relations=payload.get("relations") or [],
    )


def _get_or_create(name: str, type_: str, aliases: list[str] | None, cache: dict):
    """按归一化名找实体，找不到就建。

    ``cache`` 是本次构建内的 dict，避免每篇论文都去查一次库。
    别名也作为键写进缓存——先出现的写法决定了节点名，后出现的别名指向同一个节点。
    """
    from ..extensions import db
    from ..models.graph import Entity

    norm = normalize(name)
    if norm in cache:
        return cache[norm]

    row = db.session.query(Entity).filter(Entity.name_norm == norm).one_or_none()
    if row is None:
        row = Entity(name=name.strip(), name_norm=norm, type=type_, aliases=aliases or [])
        db.session.add(row)
        db.session.flush()   # 取到 id，后续关系要用
    else:
        # 已有的实体补别名（新出现的写法）与缺失的类型
        merged = list({*(row.aliases or []), *(aliases or []), name.strip()})
        row.aliases = [a for a in merged if normalize(a) != row.name_norm]
        if not row.type and type_:
            row.type = type_

    cache[norm] = row
    for alias in aliases or []:
        cache.setdefault(normalize(alias), row)
    return row


def index_note(note, provider=None) -> dict:
    """抽一篇笔记的图谱并落库。返回统计。

    **先删这篇论文的旧边再写**：重新抽取时实体名会变，增量更新的结果是图上
    越来越乱。整篇重来既简单又不会错。
    """
    from ..extensions import db
    from ..models.graph import PaperEntity, Relation

    result = extract(note, provider=provider)
    if result.error:
        return {"note_id": note.id, "error": result.error, "entities": 0, "relations": 0}

    paper_id = note.paper_id
    if paper_id:
        db.session.query(Relation).filter(Relation.paper_id == paper_id).delete(
            synchronize_session=False
        )
        db.session.query(PaperEntity).filter(PaperEntity.paper_id == paper_id).delete(
            synchronize_session=False
        )

    cache: dict = {}
    created_entities: dict[str, object] = {}
    # 同一篇论文里，同一个实体可能以多种写法出现（原名 + 别名），
    # 也可能被模型列了两次。`paper_entities` 的主键是 (paper_id, entity_id)，
    # 不去重就会撞主键——实测第一次跑就撞了。
    linked: set[str] = set()

    for item in result.entities:
        name = (item.get("name") or "").strip()
        type_ = item.get("type") or "concept"
        if not is_useful(name):
            continue
        row = _get_or_create(name, type_, item.get("aliases"), cache)
        created_entities[normalize(name)] = row
        if paper_id and row.id not in linked:
            linked.add(row.id)
            db.session.add(
                PaperEntity(paper_id=paper_id, entity_id=row.id, role="mentions", count=1)
            )

    kept = 0
    for item in result.relations:
        src_name = (item.get("src") or "").strip()
        dst_name = (item.get("dst") or "").strip()
        rel_type = item.get("type") or ""
        evidence = (item.get("evidence") or "").strip()
        # **没有证据的边直接丢。** 这是这套图谱唯一的价值主张，
        # 在入口处守住，比事后在界面上标「未验证」便宜得多。
        if not (src_name and dst_name and rel_type and evidence):
            continue
        if not (is_useful(src_name) and is_useful(dst_name)):
            continue

        src = created_entities.get(normalize(src_name)) or _get_or_create(
            src_name, "concept", None, cache
        )
        dst = created_entities.get(normalize(dst_name)) or _get_or_create(
            dst_name, "concept", None, cache
        )
        if src.id == dst.id:
            continue

        existing = (
            db.session.query(Relation)
            .filter(
                Relation.src_id == src.id,
                Relation.dst_id == dst.id,
                Relation.type == rel_type,
            )
            .one_or_none()
        )
        if existing is not None:
            # 同一条边被多篇论文支持时累加权重并追加证据，而不是插重复边
            existing.weight = (existing.weight or 1.0) + 1.0
            existing.evidence = [
                *(existing.evidence or []),
                {"note_id": note.id, "paper_id": paper_id, "quote": evidence[:500]},
            ]
            continue

        db.session.add(
            Relation(
                src_id=src.id,
                dst_id=dst.id,
                type=rel_type,
                paper_id=paper_id,
                weight=1.0,
                evidence=[{"note_id": note.id, "paper_id": paper_id, "quote": evidence[:500]}],
                source="ai",
            )
        )
        kept += 1

    db.session.commit()
    return {
        "note_id": note.id,
        "entities": len(created_entities),
        "relations": kept,
    }


def build(limit: int = 0, ctx=None) -> dict:
    """给所有有笔记的论文建图。"""
    from ..extensions import db
    from ..models import Note

    notes = (
        db.session.query(Note)
        .filter(Note.paper_id.isnot(None))
        .order_by(Note.id)
        .all()
    )
    if limit:
        notes = notes[:limit]

    done = failed = 0
    for index, note in enumerate(notes, 1):
        if ctx is not None:
            ctx.check_cancelled()
            ctx.progress(index / len(notes), f"建图 {index}/{len(notes)}")
        result = index_note(note)
        if result.get("error"):
            failed += 1
        else:
            done += 1
    return {"notes": len(notes), "ok": done, "failed": failed}


def stats() -> dict:
    """图的基本规模。"""
    from ..extensions import db
    from ..models.graph import Entity, PaperEntity, Relation

    return {
        "entities": db.session.query(Entity).count(),
        "relations": db.session.query(Relation).count(),
        "paper_links": db.session.query(PaperEntity).count(),
        "with_evidence": db.session.query(Relation)
        .filter(Relation.evidence.isnot(None))
        .count(),
    }


def top_entities(limit: int = 30) -> list[dict]:
    """被最多论文提到的实体。"""
    from ..extensions import db
    from ..models.graph import Entity, PaperEntity

    rows = (
        db.session.query(Entity, db.func.count(PaperEntity.paper_id).label("n"))
        .join(PaperEntity, PaperEntity.entity_id == Entity.id)
        .group_by(Entity.id)
        .order_by(db.text("n DESC"))
        .limit(limit)
        .all()
    )
    return [
        {"id": e.id, "name": e.name, "type": e.type, "papers": int(n)}
        for e, n in rows
    ]


def entity_detail(entity_id: str) -> dict | None:
    """一个实体的全部关系与出处。"""
    from ..extensions import db
    from ..models.graph import Entity, Relation

    entity = db.session.get(Entity, entity_id)
    if entity is None:
        return None

    def edge(row, other, direction):
        return {
            "id": row.id,
            "type": row.type,
            "direction": direction,
            "other_id": other.id,
            "other_name": other.name,
            "other_type": other.type,
            "weight": row.weight,
            "evidence": row.evidence or [],
        }

    out = [
        edge(r, r.dst, "out")
        for r in db.session.query(Relation).filter(Relation.src_id == entity_id).all()
    ]
    incoming = [
        edge(r, r.src, "in")
        for r in db.session.query(Relation).filter(Relation.dst_id == entity_id).all()
    ]
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type,
        "aliases": entity.aliases or [],
        "out": out,
        "in": incoming,
    }


__all__ = [
    "PROMPT_VERSION",
    "Extraction",
    "build",
    "entity_detail",
    "extract",
    "index_note",
    "is_useful",
    "normalize",
    "stats",
    "top_entities",
]
