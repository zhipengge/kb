"""深度阅读用的提示词与结构化 schema。

**提示词带版本号**，并参与阶段缓存的指纹计算。这是刻意的：
改提示词之后，旧产物会自动失效并重跑，不需要手动清缓存；
而如果提示词改了却不重跑，你看到的是新旧提示词混出来的结果，
却以为是在评估新版本——那种困惑很难排查。

改提示词的流程：改内容 + 升版本号。
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# 版本号。改动任何提示词正文时都要升，否则缓存不会失效。
#
# 这一版的方法论借用了两个开源 skill 的做法（MIT 许可），并按其结构改写：
#   * LuoHaomin/read-paper  —— 按论文类型分诊后路由到不同模板；
#     「防幻觉三闸」（溯源闸 / 重建闸 / 不补全闸）
#   * Kingslayer-bot/paper-reading-skill —— 交叉矛盾必须保留而不是抹平；
#     交付前必须过质量自检
# 它们是为 Claude Code 写的交互式 skill，这里把方法论移植成服务端的
# 结构化抽取——保留了闸门与分诊，去掉了多智能体外呼（对 98 篇批量处理
# 成本太高）。
PROMPT_VERSION = "2026-10-07.1"


# --------------------------------------------------------------------------
# 论文分诊
# --------------------------------------------------------------------------

TYPE_METHOD = "method"      # 实验方法型：提出方法 + 实证
TYPE_THEORY = "theory"      # 理论证明型：定理 / 引理 / 证明
TYPE_SURVEY = "survey"      # 综述型：分类与梳理
TYPE_SYSTEM = "system"      # 系统型：架构设计与评测

PAPER_TYPES = (TYPE_METHOD, TYPE_THEORY, TYPE_SURVEY, TYPE_SYSTEM)

_TYPE_LABELS = {
    TYPE_METHOD: "实验方法",
    TYPE_THEORY: "理论证明",
    TYPE_SURVEY: "综述",
    TYPE_SYSTEM: "系统",
}


def classify_paper(*, title: str = "", abstract: str = "", sample_text: str = "") -> str:
    """判断论文类型。

    用**结构信号**而不是让模型判断——这些信号在 LaTeX 源码里是确定存在的，
    比模型读完摘要再猜便宜且可靠得多。这也是 read-paper 那张判定信号表的实现。

    分诊的意义在于：综述要的是分类树，理论论文要的是定理与证明，
    方法论文要的是方法与实验——用同一套模板套所有论文，结果是
    哪一类都写不好。
    """
    haystack = f"{title}\n{abstract}\n{sample_text[:60000]}"
    lowered = title.lower() + " " + abstract[:600].lower()

    # 综述：标题/摘要里直接写着
    if re.search(r"\b(survey|review|overview|taxonomy|systematic review)\b", lowered):
        return TYPE_SURVEY
    # 中文标题也可能
    if re.search(r"综述|回顾|研究进展|系统性梳理", title):
        return TYPE_SURVEY

    # 理论证明：正文里有成规模的定理与证明
    theorems = len(re.findall(r"\\begin\{(theorem|lemma|proposition|corollary|proof)\}", haystack))
    theorem_cmds = len(re.findall(r"\\(theorem|lemma|proposition|corollary)\b", haystack))
    proofs = len(re.findall(r"\\begin\{proof\}|\\proof\b|\\qed\b", haystack))
    if theorems >= 3 or (theorem_cmds >= 5 and proofs >= 3):
        return TYPE_THEORY

    # 系统：标题带 System/Architecture 且有实现章节
    if re.search(r"\b(system|architecture|framework)\b", lowered) and re.search(
        r"\\section\{[^}]*[Ii]mplementation", haystack
    ):
        return TYPE_SYSTEM

    return TYPE_METHOD


# --------------------------------------------------------------------------
# 防幻觉闸门
#
# 这三个闸门来自 LuoHaomin/read-paper 的「防幻觉三闸」。原 skill 的核心判断是
# 「写之前先立规矩，比方法更重要」——因为一旦让模型开始自由发挥，
# 事后再检查已经晚了。所以它们被写进**系统提示词的最前面**，
# 而不是作为注意事项放在末尾。
# --------------------------------------------------------------------------

HALLUCINATION_GATES = """
## 硬性规则（违反任何一条，这份笔记就是废的）

**溯源闸。** 每个数字、每个指标、每条非平凡的结论，都必须能挂回原文出处：
写清它出自哪一节，或者直接引用原文的那句话。**挂不上出处的，标 `⚠️ 待核：…`，
不得当作事实写出来。**

**不补全闸。** 论文没有讨论的内容，就写「原文未讨论」。
**绝对禁止替作者编造局限、数字、引用或未来工作。**
「局限」一栏宁可空着，也不要写一个作者没说过、你以为他应该有的局限。

**矛盾保留闸。** 如果论文里的不同部分互相矛盾，或者你发现数据和结论对不上，
**指出来**。不要擅自选一个说法把它抹平——矛盾之处恰恰是最值得研究者注意的地方。
"""


MERMAID_GUIDE = """
## 图示要求

笔记里至少要有**一张 Mermaid 图**，画清这篇论文最关键的结构：
方法型画推理/训练流程，系统型画模块架构，综述型画分类树，理论型画证明骨架。

写 Mermaid 时的硬性要求（违反会导致图渲染不出来）：

- 用 `flowchart TD`（自上而下）或 `graph LR`（从左到右），不要用其它图类型。
- **节点标签里不要出现圆括号、方括号、花括号**——Mermaid 会把这些当成语法。
  数学记号要换写法：下标写成 `x_t-1` 而不是 `x_{t-1}`，上标写成 `x^2` 而不是 `x^{2}`。
  标签控制在 12 个字以内。
- 节点 id 用 `A`、`B`、`C` 这样的短标识，中文写在标签里：`A[输入论文] --> B[编码器]`。
- 只画论文里**确实描述过**的模块与数据流。**不要为了图好看而补上论文没提的组件**——
  一张编出来的架构图比没有图有害得多，因为它看起来确定、读者不会去核对。
- 分支条件是真实存在的才画（比如「训练时 / 推理时」），不要造。
"""


READ_SYSTEM = (
    """你在帮一位研究者精读论文，产出的是**他自己的研究笔记**，不是摘要。

写笔记的要求：

- 用中文写，但技术术语保留英文原文（写「注意力机制（attention）」而不是硬译）。
- 数字、指标、超参要精确引用原文，不要模糊成「显著提升」。
- 面向「三个月后回看的自己」：写清楚**为什么**这么做，而不只是做了什么。
- 每一条都要能回答「这为什么重要」——只会复述论文的笔记，不如直接看原文。
"""
    + HALLUCINATION_GATES
    + MERMAID_GUIDE
)

# 各类型共有的字段：不论什么论文都要回答的问题
_COMMON_PROPERTIES = {
    "one_liner": {
        "type": "string",
        "description": "一句话说清这篇论文做了什么（不超过 60 字）",
    },
    "problem": {
        "type": "string",
        "description": "它要解决什么问题、为什么这个问题重要",
    },
    "why_nontrivial": {
        "type": "string",
        "description": "这件事**难在哪里**——为什么不是显然就能做到的。"
        "没有这一节，笔记就退化成复读",
    },
    "key_techniques": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "note": {"type": "string", "description": "这个技术在这里怎么用的"},
            },
            "required": ["name", "note"],
            "additionalProperties": False,
        },
        "description": "用到或提出的关键技术，3-8 个",
    },
    "limitations": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source": {
                    "type": "string",
                    "enum": ["paper", "reviewer"],
                    "description": "paper=论文自己承认的；reviewer=你从方法看出的。"
                    "论文没讨论就留空数组，不要编",
                },
            },
            "required": ["text", "source"],
            "additionalProperties": False,
        },
    },
    "open_questions": {
        "type": "array",
        "items": {"type": "string"},
        "description": "读完仍存疑的地方，或值得跟进的方向。没有就留空数组",
    },
}




def _diagram_property(what: str) -> dict:
    return {
        "type": "string",
        "description": (
            f"{what}的 Mermaid 图。**只写 Mermaid 代码本身**，"
            "不要包 ```mermaid 代码围栏，不要写任何解释文字。"
        ),
    }


def _results_schema() -> dict:
    """实验结果。带上出处，让每个数字都能核对。"""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "metric": {"type": "string"},
                "value": {"type": "string", "description": "数值，含对比基准"},
                "locator": {
                    "type": "string",
                    "description": "出处：章节名 / 表号 / 页码。挂不上就留空并在 "
                    "value 里标 ⚠️ 待核",
                },
            },
            "required": ["dataset", "metric", "value"],
            "additionalProperties": False,
        },
        "description": "关键实验结果。数值要精确，不要写「效果更好」",
    }


# 按类型定制的字段。这是分诊的意义所在——
# 综述要的是分类脉络，理论论文要的是定理与证明思路，
# 方法论文要的是方法细节与实验，系统论文要的是架构与权衡。
_TYPE_SPECIFIC: dict[str, dict] = {
    TYPE_METHOD: {
        "contributions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "论文的主要贡献，逐条列出",
        },
        "method": {
            "type": "string",
            "description": "方法的核心思路。要说清关键设计选择与它们各自的理由",
        },
        "method_diagram": _diagram_property("方法流程（训练与推理的步骤、模块之间的数据流）"),
        "results": _results_schema(),
        "reproduction_notes": {
            "type": "string",
            "description": "如果要复现，哪些细节是关键、哪些从论文里看不出来。"
            "看不出来的地方直说，不要推测",
        },
    },
    TYPE_THEORY: {
        "contributions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "主要的数学结论，逐条列出",
        },
        "main_theorem": {
            "type": "string",
            "description": "核心定理的陈述（可以用数学记号）",
        },
        "proof_idea": {
            "type": "string",
            "description": "证明的核心思路：关键引理是什么、各步之间怎么衔接。"
            "**不要写「证明见附录」就跳过**——要说清证明的骨架",
        },
        "proof_diagram": _diagram_property("证明骨架（哪些引理支撑哪个结论、推导的依赖关系）"),
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "定理成立所依赖的假设。这是理论工作最需要被审视的地方",
        },
        "results": _results_schema(),
    },
    TYPE_SURVEY: {
        "contributions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "综述本身的贡献（提出了什么分类框架、整理了哪些脉络）",
        },
        "taxonomy": {
            "type": "string",
            "description": "它的分类框架是什么、按什么维度划分",
        },
        "taxonomy_diagram": _diagram_property("分类框架的层级树"),
        "key_works": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string", "description": "归属于分类框架的哪一支"},
                    "note": {"type": "string", "description": "它的贡献与地位"},
                },
                "required": ["name", "category", "note"],
                "additionalProperties": False,
            },
            "description": "综述里梳理的关键工作，5-15 个",
        },
        "open_challenges": {
            "type": "array",
            "items": {"type": "string"},
            "description": "综述指出的开放问题与未来方向",
        },
    },
    TYPE_SYSTEM: {
        "contributions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "系统的主要贡献，逐条列出",
        },
        "architecture": {
            "type": "string",
            "description": "系统的整体架构与各模块职责",
        },
        "architecture_diagram": _diagram_property("系统架构（模块划分与它们之间的依赖）"),
        "design_tradeoffs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "description": "做了什么设计选择"},
                    "rationale": {"type": "string", "description": "作者给的理由"},
                    "cost": {"type": "string", "description": "这个选择的代价或限制"},
                },
                "required": ["decision", "rationale"],
                "additionalProperties": False,
            },
            "description": "关键设计取舍。系统论文的价值主要在这里",
        },
        "results": _results_schema(),
    },
}


def schema_for(paper_type: str) -> dict:
    """按论文类型取抽取 schema。"""
    specific = _TYPE_SPECIFIC.get(paper_type, _TYPE_SPECIFIC[TYPE_METHOD])
    properties = {**_COMMON_PROPERTIES, **specific}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def type_label(paper_type: str) -> str:
    """论文类型的中文名。

    导出成公共函数是为了让渲染层复用这份映射——否则类型名会在
    提示词、渲染、前端各写一份，改名时必然漏掉一两处。
    """
    return _TYPE_LABELS.get(paper_type, paper_type)


def instructions_for(paper_type: str) -> str:
    """按类型给出补充要求。"""
    label = type_label(paper_type)
    base = (
        f"这是一篇**{label}**类论文，按对应的字段填写。"
        "请调用 submit_result 工具提交精读笔记。"
    )
    extra = {
        TYPE_THEORY: "理论论文尤其要写清假设与证明骨架——「证明见附录」不是答案。",
        TYPE_SURVEY: "综述要写清它的分类框架，以及每项工作在整个脉络里的位置。",
        TYPE_SYSTEM: "系统论文的价值在设计取舍，每条取舍都要写清代价。",
        TYPE_METHOD: "方法论文要写清为什么这样设计，而不只是做了什么。",
    }
    return f"{base} {extra.get(paper_type, '')} {HALLUCINATION_GATES}"


# 兼容旧调用点
READ_SCHEMA = schema_for(TYPE_METHOD)
READ_INSTRUCTIONS = instructions_for(TYPE_METHOD)


# --------------------------------------------------------------------------
# 打标签
# --------------------------------------------------------------------------

TAG_SYSTEM = """你在给论文打标签，用于一个**受控词表**的知识库。

规则：
- 标签用中文，专有名词保留英文（如「扩散模型（diffusion model）」）。
- 每个标签必须落在给定维度之一，不要自创新的维度。
- 优先复用已有标签：下面会给出词表里已有的标签，能对应上就用它的**原名**。
- 宁少勿多。只打有实质信息量的标签，3-6 个足够。
  打一堆「深度学习」「人工智能」这种放在任何论文上都成立的大词，等于没打。"""

TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dimension": {
                        "type": "string",
                        "enum": ["topic", "method", "task", "domain"],
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0-1，你有多确信这个标签准确",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "为什么打这个标签，一句话",
                    },
                },
                "required": ["name", "dimension", "confidence", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tags"],
    "additionalProperties": False,
}

TAG_INSTRUCTIONS = "请调用 submit_result 工具提交标签。3-6 个，宁少勿多。"


# --------------------------------------------------------------------------
# 提示词组装
# --------------------------------------------------------------------------


def build_paper_context(
    *,
    title: str,
    abstract: str = "",
    sections: list[dict] | None = None,
    max_chars: int = 24000,
    paper_type: str = TYPE_METHOD,
) -> str:
    """组装喂给模型的论文正文。

    **不是把全文塞进去。** 一篇论文动辄三四万 token，全塞进去既贵又会让
    模型在细节里迷路。这里按「精读需要什么」挑选：

      * 摘要 —— 全局定位
      * 引言与结论 —— 动机与总结
      * 方法章 —— 核心内容，给足篇幅
      * 实验章的开头 —— 主要结果

    返回的文本**顺序固定**（不随调用变化），这样才能命中提示缓存——
    缓存是前缀匹配，任何字节级差异都会让它失效。
    """
    parts: list[str] = [f"# {title}"]

    if abstract:
        parts.append(f"## 摘要\n{abstract.strip()}")

    if sections:
        budget = max_chars
        # 按「精读价值」给各章节排优先级，而不是按原始顺序平铺
        # 章节的优先级**随论文类型变化**：
        # 综述的价值在分类梳理，理论论文的价值在定理与证明，
        # 方法论文的价值在方法与实验。用同一套权重会把关键章节挤掉。
        priority_words = {
            TYPE_METHOD: {
                "method": 3.0, "approach": 3.0, "model": 2.5, "architecture": 2.5,
                "introduction": 2.0, "conclusion": 1.8, "experiment": 1.6,
                "result": 1.5, "abstract": 0.0, "related work": 0.4,
                "reference": 0.0, "acknowledg": 0.0, "appendix": 0.3,
            },
            TYPE_THEORY: {
                "theorem": 3.5, "proof": 3.5, "lemma": 3.0, "proposition": 2.8,
                "preliminar": 2.5, "main result": 3.0, "method": 2.0,
                "introduction": 1.8, "conclusion": 1.5, "experiment": 1.0,
                "abstract": 0.0, "related work": 0.5, "reference": 0.0,
                "acknowledg": 0.0, "appendix": 1.5,
            },
            TYPE_SURVEY: {
                "taxonomy": 3.5, "categor": 3.0, "classification": 3.0,
                "overview": 2.5, "introduction": 2.2, "challenge": 2.5,
                "future": 2.5, "conclusion": 2.0, "method": 1.2,
                "abstract": 0.0, "reference": 0.0, "acknowledg": 0.0,
            },
            TYPE_SYSTEM: {
                "system": 3.0, "architecture": 3.2, "design": 2.8,
                "implementation": 2.5, "evaluation": 2.2, "introduction": 1.8,
                "conclusion": 1.6, "experiment": 1.4, "abstract": 0.0,
                "related work": 0.4, "reference": 0.0, "acknowledg": 0.0,
            },
        }.get(paper_type, {})

        def weight(section: dict) -> float:
            path = (section.get("path") or section.get("title") or "").lower()
            for word, value in priority_words.items():
                if word in path:
                    return value
            return 1.0

        ranked = sorted(sections, key=weight, reverse=True)
        chosen: list[dict] = []
        for section in ranked:
            text = (section.get("text") or "").strip()
            if not text or weight(section) <= 0:
                continue
            cost = len(text)
            if cost > budget:
                text = text[:budget]
                cost = budget
            if cost <= 0:
                continue
            chosen.append({"path": section.get("path") or section.get("title"), "text": text})
            budget -= cost
            if budget <= 0:
                break

        # 按原始顺序输出。这一点不能省：提示缓存是**前缀匹配**，
        # 章节顺序若随优先级排序而变，两篇不同的调用会得到不同的前缀，
        # 缓存永远命不中。按原文顺序输出，同一篇论文的组装结果就是稳定的。
        position = {
            (section.get("path") or section.get("title")): index
            for index, section in enumerate(sections)
        }
        chosen.sort(key=lambda item: position.get(item["path"], len(sections)))

        for section in chosen:
            parts.append(f"## {section['path']}\n{section['text']}")

    return "\n\n".join(parts)


__all__ = [
    "HALLUCINATION_GATES",
    "PAPER_TYPES",
    "PROMPT_VERSION",
    "READ_INSTRUCTIONS",
    "READ_SCHEMA",
    "READ_SYSTEM",
    "TAG_INSTRUCTIONS",
    "TAG_SCHEMA",
    "TAG_SYSTEM",
    "TYPE_METHOD",
    "TYPE_SURVEY",
    "TYPE_SYSTEM",
    "TYPE_THEORY",
    "build_paper_context",
    "classify_paper",
    "instructions_for",
    "schema_for",
    "type_label",
]
