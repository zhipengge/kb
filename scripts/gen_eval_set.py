#!/usr/bin/env python
"""从语料生成检索评测集。

**为什么需要它。** 原先的评测集只有 10 道题（5 点名 + 5 概念），而且点名项
在 3/5~5/5 之间随机波动——那是查询扩展的随机性。10 道题上的 ±1 波动意味着
**任何调优都无法被证伪**：改好改坏都落在这个噪声带宽里。

**为什么不让模型自由发挥。** 金标准是「这道题该命中哪篇论文」，这一点是确定的
（题就是从这篇论文生成的）。模型只负责把论文写成一个**真实的问法**，
不负责判断答案对不对——所以这个评测仍然是客观的。

两个硬约束写进提示词，否则生成出来的题没有区分度：
  * **不许出现论文标题里的词**。否则检索只要字符串匹配就能命中，
    测的是「标题在不在索引里」，不是语义检索。
  * **问一个具体的技术点**，不要问「这篇讲了什么」这种万金油。

用法::

    pipenv run python scripts/gen_eval_set.py            # 生成
    pipenv run python scripts/gen_eval_set.py --limit 5  # 先试几篇
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb import create_app

OUT = Path(__file__).resolve().parent / "eval_questions.json"

SYSTEM = """你在为一个论文知识库写检索评测题。

给定一篇论文的标题与摘要，写 2 道**研究者真的会去搜**的问题：
  1. 一道「概念类」：描述一个技术需求或现象，答案指向这篇论文的做法。
  2. 一道「细节类」：问一个只有这篇论文会具体回答的技术细节。

**硬性要求**（违反则这条评测题作废）：
  * **绝对不能出现标题里的实词**。标题里的方法名、缩写、独特术语都不许用。
    评测的是语义检索，能靠字符串匹配命中的题没有区分度。
  * 用提问者会用的说法，不要复述摘要的句子。
  * 一道题一句话，不要「以及」「还有」并列多个问题。
  * 输出 JSON 数组，每项 {"q": "...", "kind": "concept"|"detail"}，不要有其它内容。"""


def _parse_items(text: str) -> list[dict]:
    """从模型输出里尽力捞出题目对象。

    **不指望它每次都吐出合法 JSON。** 实测这个端点会把输出截断在句子中间
    （思考与正文共用 max_tokens），`json.loads` 直接报
    ``Unterminated string``。整批丢掉太浪费——那篇论文的题其实已经写出来了，
    只是尾巴被切掉。

    所以退一步：先用正规解析，不行就按对象逐个抓。抓不到的（最后一个被截断的
    对象）自然就少一条，不影响前面那些。
    """
    text = (text or "").strip()
    if text.startswith("```"):
        # 去掉 ```json 围栏
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass

    items: list[dict] = []
    # 逐个匹配完整的 {"q": "...", "kind": "..."}，最后那个被截断的自然匹配不上
    for match in re.finditer(
        r'\{\s*"q"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"kind"\s*:\s*"(\w+)"\s*\}', text
    ):
        question = match.group(1).encode().decode("unicode_escape", errors="ignore")
        items.append({"q": question, "kind": match.group(2)})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="生成检索评测集")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（试跑用）")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        from kb.extensions import db
        from kb.models import Paper
        from kb.services import budget
        from kb.services.llm import fast_provider

        papers = (
            db.session.query(Paper)
            .filter(Paper.deleted_at.is_(None), Paper.abstract.isnot(None))
            .order_by(Paper.id)
            .all()
        )
        if args.limit:
            papers = papers[: args.limit]

        provider = fast_provider()
        cases: list[dict] = []
        failed = 0

        for index, paper in enumerate(papers, 1):
            abstract = (paper.abstract or "").strip()[:1500]
            if len(abstract) < 120:
                continue
            prompt = f"标题：{paper.title}\n\n摘要：{abstract}"
            try:
                with budget.track("eval"):
                    response = provider.complete(
                        [{"role": "user", "content": prompt}],
                        system=SYSTEM,
                        # 给足：这个端点的思考与正文**共用**这个额度，实测
                        # 思考动辄上万字符。给 2000 时输出会被截在半句上，
                        # 拿到的是残缺 JSON（实测 3 篇里坏 1 篇）。
                        max_tokens=8000,
                    )
                items = _parse_items(response.text)
            except Exception as exc:
                failed += 1
                print(f"  [{index}/{len(papers)}] 失败 {type(exc).__name__}: {exc}"[:100])
                continue

            title_words = {
                w.lower()
                for w in __import__("re").findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", paper.title or "")
            }
            kept = 0
            for item in items if isinstance(items, list) else []:
                question = (item.get("q") or "").strip()
                if not question:
                    continue
                # 硬约束的机器校验：标题实词出现在问题里就丢掉
                q_words = {
                    w.lower()
                    for w in __import__("re").findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", question)
                }
                if q_words & title_words:
                    continue
                cases.append(
                    {
                        "question": question,
                        "expect": paper.title,
                        "kind": item.get("kind") or "concept",
                    }
                )
                kept += 1
            print(f"  [{index}/{len(papers)}] {paper.title[:38]:40} 采 {kept} 条")

        Path(args.out).write_text(
            json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\n共 {len(cases)} 道题 → {args.out}（失败 {failed} 篇）")


if __name__ == "__main__":
    main()
