"""笔记待核项的抽取与消费。

**为什么需要它。** 精读笔记是 AI 生成的，结尾印着「内容为草稿，需人工核对后再
采信」——而 AI 自己也确实标出了不确定的地方（形如 ``⚠️ 待核：…``）。问题是这些
标记**散在 85 个 markdown 文件里**：要处理就得逐个打开，等于没有。

实测全库有 270 处。它们的价值差别很大，**必须分类**，否则屏幕上一半是我们自己
管线的问题、一半是论文真有问题，人很快就不看了：

  * ``内容不可得`` —— 提取没拿到（表格/图片/小节没进上下文）。**这是我们这边的账**。
  * ``论文疑点``   —— 表里数字与正文对不上、定理方向可疑、基准归属含混。
                    **这才是真正需要人判断的**，也最难自动判定。
  * ``复现阻塞``   —— 关键超参没给、评测协议没写清。做复现时必踩。
  * ``其它``

分类只做**粗筛**，用关键词而不是模型——它决定的是「先看哪一堆」，
判错了代价很小；而每跑一次都调模型，代价就大了。真正的判断留给人。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# 标记的开头。笔记里的写法是 `⚠️ 待核` / `⚠️ 待核：`，允许中间有空格与冒号。
_MARK = re.compile(r"⚠️")
# 小节标题，用来告诉人「这处在讲哪一部分」
_HEADING = re.compile(r"^(#{2,3})\s+(.+?)\s*$")

# 分类关键词。顺序即优先级——先判「论文自身的问题」，
# 因为「数值对不上」这类句子里往往也提到「正文/表格」，先判可忽略会把它们误吞。
_PAPER_ISSUE = re.compile(
    r"(对不上|矛盾|不吻合|不一致|缺(?:乏)?证据|反直觉|有偏|偏差|"
    r"归属|未说明|未明确|歧义|应(?:该)?落|差\s*[\d.]+|不成立|可疑|张力)"
)
_REPRO = re.compile(r"(超参|复现必需|未给出.{0,6}(?:取值|数值|配置)|参数未|配置未|复现时)")
_UNAVAILABLE = re.compile(
    r"(提供|给定|摘录|截断|原文|正文|文本|转录|排版|图表?未|表号|Tab\.|Table|"
    r"图\s*\[|<ref>|未含|未见|未出现|无法核对|不可得|节选|缺失)"
)

KIND_PAPER = "paper"      # 论文自身的疑点
KIND_REPRO = "repro"      # 复现阻塞
KIND_UNAVAILABLE = "unavailable"   # 内容没拿到（我们管线的问题）
KIND_OTHER = "other"

KIND_LABELS = {
    KIND_PAPER: "论文疑点",
    KIND_REPRO: "复现阻塞",
    KIND_UNAVAILABLE: "内容未取到",
    KIND_OTHER: "其它",
}
# 展示顺序：要人判断的排前面，我们自己管线的账排后面
KIND_ORDER = [KIND_PAPER, KIND_REPRO, KIND_OTHER, KIND_UNAVAILABLE]


@dataclass
class Flag:
    """一条待核项。"""

    note_id: str
    paper_id: str | None
    paper_title: str
    note_title: str
    note_version: int
    section: str          # 所属小节标题
    text: str             # 标记所在行的正文（去掉 ⚠️ 与列表符号）
    kind: str
    # 处理状态由 collect 之后填充（见 state_map），默认未处理
    state: str = ""
    comment: str = ""

    @property
    def flag_id(self) -> str:
        """稳定标识：同一篇笔记里内容相同的标记得到同一个 id。

        用它记录「已处理」——按出现位置编号不行，笔记一重生成位置全变。
        内容哈希的代价是：改写过的同一处会被当成新项，需要重新看。
        **这个方向是安全的**：宁可让人多看一眼，也不要把一个改过措辞的问题
        悄悄当成已处理。
        """
        raw = f"{self.note_id}\x00{self.section}\x00{self.text}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "flag_id": self.flag_id,
            "note_id": self.note_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "note_title": self.note_title,
            "note_version": self.note_version,
            "section": self.section,
            "text": self.text,
            "kind": self.kind,
            "kind_label": KIND_LABELS.get(self.kind, self.kind),
            "state": self.state,
            "comment": self.comment,
        }


def classify(text: str) -> str:
    """粗分类。顺序不能反，见模块头部的说明。"""
    if _PAPER_ISSUE.search(text):
        return KIND_PAPER
    if _REPRO.search(text):
        return KIND_REPRO
    if _UNAVAILABLE.search(text):
        return KIND_UNAVAILABLE
    return KIND_OTHER


def _clean(line: str) -> str:
    """去掉列表符号与标记本身，留下人要看的那句话。"""
    text = line.strip()
    text = re.sub(r"^[-*+]\s+", "", text)
    text = re.sub(r"^\[[ xX]\]\s*", "", text)
    # 标记可能出现在行首、行中或行尾，统一删掉标记词本身
    text = re.sub(r"⚠️\s*待核[：:]?", "", text)
    return text.strip(" -—·\t")


def extract_from_note(note) -> list[Flag]:
    """从一篇笔记里抽出所有待核项。

    按行扫，同时跟踪当前所处的小节——「哪一节里的问题」是这个列表能用的前提，
    否则一堆孤立的句子没法判断轻重。
    """
    body = note.content_md or ""
    if "⚠️" not in body:
        return []

    paper_title = ""
    paper = getattr(note, "paper", None)
    if paper is not None:
        paper_title = paper.title or ""

    flags: list[Flag] = []
    section = "（前言）"
    for line in body.splitlines():
        heading = _HEADING.match(line)
        if heading:
            section = heading.group(2).strip()
            continue
        if not _MARK.search(line):
            continue
        text = _clean(line)
        if not text:
            continue
        flags.append(
            Flag(
                note_id=note.id,
                paper_id=note.paper_id,
                paper_title=paper_title,
                note_title=note.title or "",
                note_version=note.version or 0,
                section=section,
                text=text,
                kind=classify(text),
            )
        )
    return flags


def collect(*, kinds: list[str] | None = None, note_id: str | None = None) -> list[Flag]:
    """全库收集待核项。"""
    from ..extensions import db
    from ..models import Note

    query = db.session.query(Note)
    if note_id:
        query = query.filter(Note.id == note_id)

    flags: list[Flag] = []
    for note in query.all():
        flags.extend(extract_from_note(note))

    if kinds:
        flags = [f for f in flags if f.kind in kinds]
    flags.sort(key=lambda f: (KIND_ORDER.index(f.kind), f.paper_title, f.section))
    return flags


STATE_DONE = "done"
STATE_IGNORED = "ignored"


def state_map(flag_ids: list[str] | None = None) -> dict[str, dict]:
    """取出已处理的状态，按 flag_id 索引。

    一次性取完而不是逐条查：这个页面一屏就有几百条，逐条查是几百次往返。
    """
    from ..extensions import db
    from ..models import NoteFlagState

    query = db.session.query(NoteFlagState)
    if flag_ids:
        query = query.filter(NoteFlagState.flag_id.in_(flag_ids))
    return {
        row.flag_id: {
            "state": row.state,
            "comment": row.comment or "",
            "note_version": row.note_version,
        }
        for row in query.all()
    }


def mark(
    flag_id: str,
    note_id: str,
    *,
    note_version: int = 0,
    state: str = STATE_DONE,
    comment: str = "",
) -> None:
    """标记一条待核项已处理。幂等——重复标记只更新。"""
    from ..extensions import db
    from ..models import NoteFlagState

    row = (
        db.session.query(NoteFlagState)
        .filter(NoteFlagState.flag_id == flag_id)
        .one_or_none()
    )
    if row is None:
        row = NoteFlagState(flag_id=flag_id, note_id=note_id)
        db.session.add(row)
    row.state = state if state in (STATE_DONE, STATE_IGNORED) else STATE_DONE
    row.comment = (comment or "")[:1024] or None
    # 记下处理时的笔记版本：笔记被重写后，界面上能提示「这条是在旧版上处理的」
    row.note_version = note_version
    db.session.commit()


def unmark(flag_id: str) -> None:
    """撤销处理状态（标错了要能改回来）。"""
    from ..extensions import db
    from ..models import NoteFlagState

    db.session.query(NoteFlagState).filter(NoteFlagState.flag_id == flag_id).delete(
        synchronize_session=False
    )
    db.session.commit()


def summary(flags: list[Flag] | None = None) -> dict:
    """按类别与论文汇总，供页面顶部显示。"""
    from collections import Counter

    flags = flags if flags is not None else collect()
    by_kind = Counter(f.kind for f in flags)
    by_paper = Counter(f.paper_title or f.note_title for f in flags)
    return {
        "total": len(flags),
        "by_kind": {k: by_kind.get(k, 0) for k in KIND_ORDER},
        "papers": len({f.note_id for f in flags}),
        "top_papers": by_paper.most_common(8),
    }


__all__ = [
    "KIND_LABELS",
    "KIND_ORDER",
    "STATE_DONE",
    "STATE_IGNORED",
    "Flag",
    "classify",
    "collect",
    "extract_from_note",
    "mark",
    "state_map",
    "summary",
    "unmark",
]
