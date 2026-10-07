#!/usr/bin/env python
"""检索质量评测：用固定的问题集量 top-1 / top-3 命中率。

**为什么需要它。** 检索调优最容易犯的错是「拿一两个案例反复试」——
改一个参数，某个案例好了，另一个坏了，来回拉锯，最后不知道整体是变好
还是变坏。更隐蔽的是：查询扩展走 LLM，**同一句话两次扩展可能不同**，
所以两次运行的差异里混着随机噪声，肉眼根本分不清是改动生效还是运气。

这个脚本把评测固定下来：

  * 问题集写死在下面，覆盖「点名论文」和「描述概念」两类问法；
  * 每轮开始清一次扩展缓存，让全部查询处于同等条件；
  * 默认跑两轮并对比，**把随机噪声暴露出来**——如果两轮结果不一致，
    说明评测本身不稳定，先解决那个再谈调优。

用法::

    pipenv run python scripts/eval_retrieval.py           # 跑两轮
    pipenv run python scripts/eval_retrieval.py --rounds 1
    pipenv run python scripts/eval_retrieval.py -v        # 打印每条的来源通道
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb import create_app
from kb.services import query_expand
from kb.services.search import search

# (问题, 期望命中的论文标题片段, 类别)
#
# 两类问法的期望不同：
#   * 点名类——问题里出现了论文的缩写或独有术语，期望它排第 1；
#   * 概念类——没有特定目标，只要排前面的论文确实切题即可，
#     所以期望值填「合理即可」，用 None 表示不做断言，只看结果。
CASES: list[tuple[str, str | None, str]] = [
    ("DDPM 论文的核心贡献是什么？", "Denoising Diffusion Probabilistic Models", "点名-缩写"),
    ("VGGT 怎么做的？", "VGGT", "点名-缩写"),
    ("UniAD 是什么方法？", "Planning-oriented Autonomous Driving", "点名-展开词"),
    ("Classifier-Free 那篇讲了什么？", "Classifier-Free Diffusion Guidance", "点名-标题词"),
    ("Sparse4D v3 的改进点", "Sparse4D v3", "点名-罕见词"),
    ("扩散模型的训练目标是什么", None, "概念"),
    ("世界模型在自动驾驶里怎么用", None, "概念"),
    ("多智能体轨迹预测", None, "概念"),
    ("如何评估端到端驾驶系统", None, "概念"),
    ("变分自编码器的优化目标", None, "概念"),
]


def _load_generated() -> list[tuple[str, str | None, str]]:
    """加载 ``gen_eval_set.py`` 生成的大评测集（有就用，没有就跳过）。

    手写那 10 道题保留：它们是跨版本可比的那把尺子，不能因为有了更大的集合
    就丢掉——新集合换了题面，分数不可直接和历史的比。
    """
    path = Path(__file__).resolve().parent / "eval_questions.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [
        (row["question"], row.get("expect"), f"生成-{row.get('kind', '?')}")
        for row in rows
        if row.get("question") and row.get("expect")
    ]


def run_round(verbose: bool, cases) -> list[tuple[str, str | None, str | None, list[str]]]:
    """跑一轮，返回 [(问题, 期望, 实际 top1, top3 标题列表)]。"""
    results = []
    for question, expected, kind in cases:
        hits = search(question, limit=3)
        titles = [(h.paper_title or "") for h in hits]
        top1 = titles[0] if titles else None
        results.append((question, expected, top1, titles))
        if verbose:
            print(f"    [{kind}] {question}")
            for rank, (title, hit) in enumerate(zip(titles, hits, strict=False), 1):
                mark = "★" if expected and expected in title else " "
                print(f"      {mark}{rank}. {title[:56]:<58} {hit.sources}")
    return results


def score(results) -> dict:
    """打分。

    **概念类必须单独断言「有结果」**，不能只当成功案例的陪衬。

    早先这里只对点名类打分（``if e`` 把概念类整个滤掉了），而这恰好是
    最难的一类：中文长问句在 trigram 下整句匹配不到任何东西，
    英文全文通道又够不着中文。实测「扩散策略怎么做动作去噪」在查询扩展
    不可用时返回 **0 条**——用户在界面上看到的是「知识库里没有找到相关
    内容」，而真相是检索降级了。一个不被断言的用例永远发现不了这种事。
    """
    named = [(q, e, t) for q, e, t, _ in results if e]
    top1 = sum(1 for _, e, t in named if t and e in t)
    top3 = sum(
        1 for q, e, t, titles in results if e and any(e in x for x in titles)
    )

    concepts = [(q, titles) for q, e, _, titles in results if not e]
    concept_hit = sum(1 for _, titles in concepts if titles)

    return {
        "named_top1": top1,
        "named_top3": top3,
        "named_total": len(named),
        "concept_hit": concept_hit,
        "concept_total": len(concepts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--rounds", type=int, default=2, help="跑几轮（默认 2，用于观察随机性）")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每条结果的来源通道")
    parser.add_argument("--hand-only", action="store_true", help="只用那 10 道手写题（与历史分数可比）")
    args = parser.parse_args()

    app = create_app()
    generated = [] if args.hand_only else _load_generated()
    cases = CASES + generated
    if generated:
        print(f"评测集：手写 {len(CASES)} 道 + 生成 {len(generated)} 道 = {len(cases)} 道")
    else:
        print(f"评测集：手写 {len(CASES)} 道（生成集不存在，跑 gen_eval_set.py 可扩容）")

    rounds = []
    with app.app_context():
        for index in range(args.rounds):
            # 清掉扩展缓存：否则第二轮直接复用第一轮的结果，
            # 看起来「完全一致」，实际是把随机性藏起来了
            query_expand.clear_cache()
            print(f"\n{'=' * 72}\n第 {index + 1} 轮", flush=True)
            rounds.append(run_round(args.verbose, cases))

    print(f"\n{'=' * 72}\n结果")
    print(f"{'轮次':<6}{'点名Top1':<12}{'点名Top3':<12}{'概念有结果':<12}")
    for index, results in enumerate(rounds, 1):
        stats = score(results)
        named_top1 = f"{stats['named_top1']}/{stats['named_total']}"
        named_top3 = f"{stats['named_top3']}/{stats['named_total']}"
        concept = f"{stats['concept_hit']}/{stats['concept_total']}"
        print(f"{index:<6}{named_top1:<12}{named_top3:<12}{concept:<12}")

    if len(rounds) > 1:
        # 逐条比对两轮结果。不一致的条目说明评测本身不稳定，
        # 这时讨论「涨了还是跌了」没有意义——先让评测可信。
        unstable = [
            (a[0], a[2], b[2])
            for a, b in zip(rounds[0], rounds[1], strict=False)
            if a[2] != b[2]
        ]
        if unstable:
            print(f"\n⚠ 两轮结果不一致的条目（共 {len(unstable)} 条）：")
            for question, first, second in unstable:
                print(f"  · {question}")
                print(f"      第一轮: {(first or '（空）')[:52]}")
                print(f"      第二轮: {(second or '（空）')[:52]}")
            print("\n  差异来自查询扩展的随机性。评测不稳定时，先固定扩展结果再谈调优。")
        else:
            print("\n两轮完全一致，评测稳定。")


if __name__ == "__main__":
    main()
