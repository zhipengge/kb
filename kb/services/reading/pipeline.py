"""深度阅读流水线的阶段编排。

每个阶段的产物按**指纹**缓存：``hash(论文内容 + 阶段名 + 提示词版本 + 模型 + 参数)``。

指纹里为什么要有提示词版本和模型：这两个一改，产物的质量就变了。
如果只按论文内容缓存，改了提示词却拿到旧产物，你会以为新提示词没效果——
实际上它根本没跑。这类「改了没反应」的问题非常难排查。
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field

from ...extensions import db
from ...models import Artifact, Chunk, Note, Paper
from ...models.base import utcnow
from .. import budget, tagging
from .prompts import (
    PROMPT_VERSION,
    READ_SYSTEM,
    TAG_INSTRUCTIONS,
    TAG_SCHEMA,
    TAG_SYSTEM,
    TYPE_METHOD,
    TYPE_SURVEY,
    TYPE_SYSTEM,
    TYPE_THEORY,
    build_paper_context,
    classify_paper,
    instructions_for,
    schema_for,
    type_label,
)

log = logging.getLogger(__name__)

# 阶段名。顺序即执行顺序。
STAGE_SUMMARIZE = "summarize"
STAGE_TAG = "tag"
STAGE_PUBLISH = "publish"
ALL_STAGES = (STAGE_SUMMARIZE, STAGE_TAG, STAGE_PUBLISH)


@dataclass
class PipelineResult:
    """一次流水线执行的结果。"""

    paper_id: str
    note_id: str | None = None
    stages_run: list[str] = field(default_factory=list)
    stages_cached: list[str] = field(default_factory=list)
    tags_created: list[str] = field(default_factory=list)
    tokens_used: int = 0
    elapsed_ms: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "note_id": self.note_id,
            "stages_run": self.stages_run,
            "stages_cached": self.stages_cached,
            "tags_created": self.tags_created,
            "tokens_used": self.tokens_used,
            "elapsed_ms": self.elapsed_ms,
            "errors": self.errors,
        }


# --------------------------------------------------------------------------
# 产物缓存
# --------------------------------------------------------------------------


def _fingerprint(paper: Paper, stage: str, model: str, extra: str = "") -> str:
    parts = [
        paper.file_hash or paper.id,
        stage,
        PROMPT_VERSION,
        model,
        extra,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _load_artifact(paper_id: str, stage: str, fingerprint: str) -> dict | None:
    row = (
        db.session.query(Artifact)
        .filter_by(paper_id=paper_id, stage=stage, fingerprint=fingerprint)
        .one_or_none()
    )
    if row is None:
        return None
    return row.payload if isinstance(row.payload, dict) else None


def _save_artifact(
    paper_id: str, stage: str, fingerprint: str, payload: dict, *, job_id: str | None = None
) -> None:
    row = (
        db.session.query(Artifact)
        .filter_by(paper_id=paper_id, stage=stage, fingerprint=fingerprint)
        .one_or_none()
    )
    if row is None:
        row = Artifact(
            paper_id=paper_id, stage=stage, fingerprint=fingerprint, payload=payload
        )
        db.session.add(row)
    else:
        row.payload = payload
    row.job_id = job_id
    db.session.commit()


def _paper_sections(paper: Paper) -> list[dict]:
    """把已索引的分块按章节聚合成「章节 -> 正文」。

    直接用分块而不是重新解析：分块时已经做过章节识别与清洗
    （剥离注释、去掉排版命令），重新解析一遍既慢又会得到不同的结果。
    """
    chunks = (
        db.session.query(Chunk)
        .filter(Chunk.paper_id == paper.id)
        .order_by(Chunk.ord)
        .all()
    )
    grouped: dict[str, list[str]] = {}
    order: list[str] = []

    for chunk in chunks:
        # 公式与图注不参与精读的上下文组装——它们会被单独引用，
        # 混进散文里只会让模型分心
        if chunk.kind in {"formula", "figure", "table"}:
            continue
        key = chunk.section_path or "正文"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(chunk.text)

    return [
        {"path": path, "text": "\n\n".join(grouped[path])}
        for path in order
    ]


# --------------------------------------------------------------------------
# 阶段实现
# --------------------------------------------------------------------------


def _stage_summarize(paper: Paper, provider, ctx) -> tuple[dict | None, str | None]:
    """精读并产出结构化笔记内容。

    先**分诊论文类型**再选模板：综述要的是分类脉络，理论论文要的是定理与
    证明骨架，方法论文要的是方法与实验。用同一套字段套所有论文，
    结果是哪一类都写不好。
    """
    sections = _paper_sections(paper)
    if not sections and not paper.abstract:
        return None, "论文还没有索引内容，无法精读"

    # 分诊用的是**结构信号**（定理环境、章节标题、标题里的 survey），
    # 不是让模型读完摘要再猜——这些信号在 LaTeX 源码里是确定存在的。
    sample = "\n".join(section.get("text", "") for section in sections[:40])
    paper_type = classify_paper(
        title=paper.title or "",
        abstract=paper.abstract or "",
        sample_text=sample,
    )

    context = build_paper_context(
        title=paper.title or "",
        abstract=paper.abstract or "",
        sections=sections,
        paper_type=paper_type,
    )
    if ctx is not None:
        ctx.progress(0.3, f"正在精读（{paper_type} 型，{len(context)} 字符上下文）")

    result = provider.extract(
        [{"role": "user", "content": context}],
        schema=schema_for(paper_type),
        description="提交论文精读的结构化笔记",
        system=READ_SYSTEM,
        instructions=instructions_for(paper_type),
        # 精读的输出很长（尤其是综述的关键工作列表），而推理型模型还会
        # 先花掉大量额度思考。这里给足，避免参数被截断成空对象。
        max_tokens=32000,
    )
    if isinstance(result, dict):
        result["_paper_type"] = paper_type
    return result, None


def _stage_tag(paper: Paper, provider, ctx, summary: dict | None) -> tuple[dict | None, str | None]:
    """提取标签。"""
    # 把已有词表给模型看，让它可以复用而不是造新词——
    # 这是「防止标签爆炸」在提示词层面的对应措施
    existing = tagging.list_tags()
    by_dimension: dict[str, list[str]] = {}
    for tag in existing:
        by_dimension.setdefault(tag.dimension, []).append(tag.name)

    vocabulary = "\n".join(
        f"- {dimension}: {', '.join(names[:40])}"
        for dimension, names in sorted(by_dimension.items())
    )

    parts = [f"论文标题：{paper.title}"]
    if paper.abstract:
        parts.append(f"摘要：{paper.abstract[:1200]}")
    if summary:
        parts.append(f"一句话总结：{summary.get('one_liner', '')}")
        contributions = summary.get("contributions") or []
        if contributions:
            parts.append("主要贡献：\n" + "\n".join(f"- {c}" for c in contributions[:5]))
        techniques = summary.get("key_techniques") or []
        if techniques:
            parts.append(
                "关键技术：\n"
                + "\n".join(f"- {t.get('name')}" for t in techniques[:8])
            )
    if vocabulary:
        parts.append(f"词表里已有的标签（能对应就用原名）：\n{vocabulary}")

    if ctx is not None:
        ctx.progress(0.6, "正在提取标签")

    result = provider.extract(
        [{"role": "user", "content": "\n\n".join(parts)}],
        schema=TAG_SCHEMA,
        description="提交论文的标签",
        system=TAG_SYSTEM,
        instructions=TAG_INSTRUCTIONS,
    )
    return result, None


def _stage_publish(
    paper: Paper, summary: dict, tags: dict | None, ctx
) -> tuple[str | None, str | None]:
    """把精读结果发布成笔记。"""
    from .. import notes as notes_service

    body = _render_note_markdown(paper, summary, tags)

    source_type = (paper.meta or {}).get("source_type")
    note = notes_service.create_note(
        title=f"{paper.title} — 精读笔记",
        content_md=body,
        paper_id=paper.id,
        kind="deep_read",
        source="ai",
        status="draft",
        model=(paper.meta or {}).get("reading_model"),
        prompt_version=PROMPT_VERSION,
        meta={
            "source_type": source_type,
            "generated_at": utcnow().isoformat(),
        },
    )

    if tags:
        for item in tags.get("tags") or []:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            tagging.attach_tag(
                note,
                name,
                dimension=item.get("dimension") or "misc",
                source="ai",
                confidence=item.get("confidence"),
            )

    if ctx is not None:
        ctx.progress(0.95, "笔记已生成")

    log.info("已为论文 %s 生成精读笔记 %s", paper.id, note.id)
    return note.id, None


def _sanitize_mermaid(code: str) -> str:
    """清掉会让 Mermaid 解析失败的字符。

    **实测：提示词里写了「标签里不要出现括号」也没用。** 模型表达下标时
    很自然地写 ``x_{t-1}``，而花括号在 Mermaid 里是节点形状的语法，
    整张图直接渲染不出来——而用户看到的只是「图没了」，不会知道原因。

    数学记号与图表语法在这里天然冲突，靠提示词约束不住，必须在渲染前处理。
    只清理**方括号标签内部**的字符：花括号在标签外是合法的菱形节点语法
    （``B{判断}``），一律删掉会把正常的判断节点也弄坏。
    """

    def clean(match: re.Match) -> str:
        inner = match.group(1)
        # 圆括号换成全角而不是删掉：`sqrt(1-a)` 删成 `sqrt1-a` 会看不懂，
        # 而全角的 `（）` 在 Mermaid 里是普通字符，视觉上几乎一样。
        inner = inner.replace("(", "（").replace(")", "）")
        # 花括号和方括号没有等价的替代写法，只能删
        for char in "{}[]":
            inner = inner.replace(char, "")
        return f"[{inner}]"

    return re.sub(r"\[([^\]]*)\]", clean, code)


def _render_note_markdown(paper: Paper, summary: dict, tags: dict | None) -> str:
    """把结构化结果渲染成 Markdown 笔记。

    渲染**按论文类型分叉**：综述的分类框架、理论的定理与假设、系统的设计取舍，
    分别是各自类型里最有价值的部分，用同一套小节套所有论文会把这部分挤没。

    这里必须和 ``prompts.schema_for`` 同步演进。只加 schema 不改本函数，
    抽出来的字段会在写盘前被**静默丢掉**——数据在 artifacts 里躺着，
    笔记里却看不到，而且不报任何错。这类不一致比抽取失败更难发现。
    """
    lines: list[str] = []
    source = (paper.meta or {}).get("source_type")
    source_label = {"latex": "LaTeX 源码", "pdf": "PDF 解析"}.get(source, "未知")
    paper_type = summary.get("_paper_type") or TYPE_METHOD

    def section(title: str, body: str) -> None:
        if body and str(body).strip():
            lines.append(f"## {title}")
            lines.append("")
            lines.append(str(body).strip())
            lines.append("")

    def mermaid(key: str, caption: str = "") -> None:
        """把抽取到的 Mermaid 图写进正文。

        模型偶尔会自己套上 ```mermaid 围栏，这里先剥掉再统一加——
        不剥的话会产出嵌套的围栏，整块代码都渲染不出来。
        """
        code = str(summary.get(key) or "").strip()
        if not code:
            return
        code = re.sub(r"^```(?:mermaid)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code).strip()
        code = _sanitize_mermaid(code)
        if not code:
            return
        if caption:
            lines.append(f"**{caption}**")
            lines.append("")
        lines.append("```mermaid")
        lines.append(code)
        lines.append("```")
        lines.append("")

    def bullets(title: str, items: list) -> None:
        cleaned = [str(x).strip() for x in (items or []) if str(x).strip()]
        if cleaned:
            lines.append(f"## {title}")
            lines.append("")
            for item in cleaned:
                lines.append(f"- {item}")
            lines.append("")

    lines.append(f"> {summary.get('one_liner', '')}")
    lines.append("")

    if tags:
        names = [item.get("name") for item in (tags.get("tags") or []) if item.get("name")]
        for name in names:
            lines.append(f"`{name}`")
        lines.append("")

    section("要解决的问题", summary.get("problem", ""))

    # 「难在哪里」是笔记和「把原文缩短一遍」的分界线。
    section("难在哪里", summary.get("why_nontrivial", ""))

    bullets("主要贡献", summary.get("contributions") or [])

    # ---- 分类型的核心内容 ----
    if paper_type == TYPE_SURVEY:
        section("分类框架", summary.get("taxonomy", ""))
        mermaid("taxonomy_diagram", "分类框架图")
        works = [w for w in (summary.get("key_works") or []) if w.get("name")]
        if works:
            lines.append("## 关键工作")
            lines.append("")
            lines.append("| 工作 | 归类 | 贡献与地位 |")
            lines.append("|---|---|---|")
            for work in works:
                cells = [
                    str(work.get(key, "")).replace("|", "\\|").replace("\n", " ")
                    for key in ("name", "category", "note")
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        bullets("开放问题", summary.get("open_challenges") or [])

    elif paper_type == TYPE_THEORY:
        section("核心定理", summary.get("main_theorem", ""))
        section("证明思路", summary.get("proof_idea", ""))
        mermaid("proof_diagram", "证明骨架")
        bullets("成立假设", summary.get("assumptions") or [])

    elif paper_type == TYPE_SYSTEM:
        section("系统架构", summary.get("architecture", ""))
        mermaid("architecture_diagram", "系统架构图")
        tradeoffs = [t for t in (summary.get("design_tradeoffs") or []) if t.get("decision")]
        if tradeoffs:
            lines.append("## 设计取舍")
            lines.append("")
            for item in tradeoffs:
                lines.append(f"**{item.get('decision', '')}**")
                lines.append("")
                if item.get("rationale"):
                    lines.append(f"- 为什么这么做：{item['rationale']}")
                if item.get("cost"):
                    lines.append(f"- 代价：{item['cost']}")
                lines.append("")

    else:
        section("方法", summary.get("method", ""))
        mermaid("method_diagram", "方法流程图")

    # ---- 各类型共有的尾段 ----
    techniques = summary.get("key_techniques") or []
    if techniques:
        lines.append("## 关键技术")
        lines.append("")
        for item in techniques:
            lines.append(f"- **{item.get('name', '')}**：{item.get('note', '')}")
        lines.append("")

    results = summary.get("results") or []
    if results:
        # 有出处的结果单列一栏。没有出处的位置留空，让「哪些数字没挂上来源」
        # 一眼可见——这是溯源闸在渲染层的落点。
        with_locator = any(item.get("locator") for item in results)
        lines.append("## 实验结果")
        lines.append("")
        lines.append("| 数据集 | 指标 | 数值 | 出处 |" if with_locator else "| 数据集 | 指标 | 数值 |")
        lines.append("|---|---|---|" + ("---|" if with_locator else ""))
        for item in results:
            cells = [
                str(item.get(key, "")).replace("|", "\\|")
                for key in ("dataset", "metric", "value")
            ]
            if with_locator:
                cells.append(str(item.get("locator") or "").replace("|", "\\|"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    limitations = summary.get("limitations") or []
    if limitations:
        lines.append("## 局限")
        lines.append("")
        paper_said = [x for x in limitations if x.get("source") == "paper"]
        reviewer = [x for x in limitations if x.get("source") != "paper"]
        if paper_said:
            lines.append("**论文承认的：**")
            lines.append("")
            for item in paper_said:
                lines.append(f"- {item.get('text', '')}")
            lines.append("")
        if reviewer:
            lines.append("**阅读时看出的：**")
            lines.append("")
            for item in reviewer:
                lines.append(f"- {item.get('text', '')}")
            lines.append("")

    if summary.get("reproduction_notes"):
        lines.append("## 复现要点")
        lines.append("")
        lines.append(summary["reproduction_notes"])
        lines.append("")

    questions = summary.get("open_questions") or []
    if questions:
        lines.append("## 待跟进")
        lines.append("")
        for item in questions:
            lines.append(f"- [ ] {item}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"*由 AI 生成，论文类型：{type_label(paper_type)}，正文来源：{source_label}，"
        f"提示词版本 {PROMPT_VERSION}。内容为草稿，需人工核对后再采信。*"
    )

    return "\n".join(lines)


# --------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------


def run_pipeline(
    paper: Paper,
    *,
    ctx=None,
    stages: tuple[str, ...] = ALL_STAGES,
    force: bool = False,
) -> PipelineResult:
    """对一篇论文跑深度阅读流水线。

    ``force=True`` 时忽略缓存全部重跑（改了阶段实现但没升版本号时用）。
    """
    from ..llm import get_provider

    started = time.perf_counter()
    result = PipelineResult(paper_id=paper.id)

    provider = get_provider()
    model = getattr(provider, "model", "unknown")
    paper.meta = {**(paper.meta or {}), "reading_model": model}

    summary: dict | None = None
    tags: dict | None = None

    # ---- 精读 ----
    if STAGE_SUMMARIZE in stages:
        fingerprint = _fingerprint(paper, STAGE_SUMMARIZE, model)
        if not force:
            summary = _load_artifact(paper.id, STAGE_SUMMARIZE, fingerprint)
        if summary is not None:
            result.stages_cached.append(STAGE_SUMMARIZE)
        else:
            if ctx is not None:
                ctx.check_cancelled()
            with budget.track("read", ref=paper.id):
                summary, error = _stage_summarize(paper, provider, ctx)
            if error:
                result.errors.append(error)
                return _finish(result, started)
            _save_artifact(paper.id, STAGE_SUMMARIZE, fingerprint, summary or {})
            result.stages_run.append(STAGE_SUMMARIZE)
            result.tokens_used += getattr(provider.last_usage, "input_tokens", 0)
            result.tokens_used += getattr(provider.last_usage, "output_tokens", 0)

    # ---- 打标 ----
    if STAGE_TAG in stages:
        fingerprint = _fingerprint(paper, STAGE_TAG, model)
        if not force:
            tags = _load_artifact(paper.id, STAGE_TAG, fingerprint)
        if tags is not None:
            result.stages_cached.append(STAGE_TAG)
        else:
            if ctx is not None:
                ctx.check_cancelled()
            with budget.track("tag", ref=paper.id):
                tags, error = _stage_tag(paper, provider, ctx, summary)
            if error:
                result.errors.append(error)
            elif tags is not None:
                _save_artifact(paper.id, STAGE_TAG, fingerprint, tags)
                result.stages_run.append(STAGE_TAG)
                result.tokens_used += getattr(provider.last_usage, "input_tokens", 0)
                result.tokens_used += getattr(provider.last_usage, "output_tokens", 0)

    # ---- 发布 ----
    if STAGE_PUBLISH in stages:
        if ctx is not None:
            ctx.check_cancelled()
        if summary is None:
            result.errors.append("没有精读结果，无法发布笔记")
            return _finish(result, started)

        # 已有同一篇论文的精读笔记时不重复创建，改为覆盖它的内容——
        # 每次重跑都新建一篇会让笔记列表迅速堆满同一篇论文的多个版本
        existing = (
            db.session.query(Note)
            .filter(Note.paper_id == paper.id, Note.kind == "deep_read")
            .order_by(Note.created_at.desc())
            .first()
        )
        body = _render_note_markdown(paper, summary, tags)

        if existing is not None:
            from .. import notes as notes_service

            notes_service.update_note(
                existing.id,
                {
                    "content_md": body,
                    "title": f"{paper.title} — 精读笔记",
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                },
                author="ai",
            )
            result.note_id = existing.id
            note = existing
        else:
            note_id, error = _stage_publish(paper, summary, tags, ctx)
            if error:
                result.errors.append(error)
                return _finish(result, started)
            result.note_id = note_id
            note = db.session.get(Note, note_id) if note_id else None

        # 标签在这一步统一附加，覆盖「笔记已存在」的情况
        if note is not None and tags:
            result.tags_created = _sync_note_tags(note, tags)

        result.stages_run.append(STAGE_PUBLISH)

    return _finish(result, started)


def _sync_note_tags(note: Note, tags: dict) -> list[str]:
    """把标签附加到笔记上，返回实际用到的标签名。"""
    names: list[str] = []
    for item in tags.get("tags") or []:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        try:
            tag = tagging.attach_tag(
                note, name,
                dimension=item.get("dimension") or "misc",
                source="ai",
                confidence=item.get("confidence"),
            )
            if tag is not None:
                names.append(tag.name)
        except Exception:
            log.warning("附加标签 %r 失败", name, exc_info=True)
    return names


def _finish(result: PipelineResult, started: float) -> PipelineResult:
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    db.session.commit()
    return result


__all__ = [
    "ALL_STAGES",
    "STAGE_PUBLISH",
    "STAGE_SUMMARIZE",
    "STAGE_TAG",
    "PipelineResult",
    "run_pipeline",
]
