#!/usr/bin/env python
"""生成演示用的假论文，用于在没有真实数据时验证整条流水线。

**刻意不包含任何真实论文**：仓库只含代码，用户的数据归用户。
这里用 PyMuPDF 现场生成几份结构完整的假 PDF（标题、作者、摘要、
章节、公式、图表、arXiv 编号、参考文献），足以覆盖解析、分块、
检索、去重等所有环节。

用法::

    pipenv run python scripts/seed_demo.py /tmp/demo-papers
    pipenv run python scripts/seed_demo.py /tmp/demo-papers --duplicates

``--duplicates`` 会额外制造重复文件与近似标题，用来验证去重逻辑。
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pymupdf

# 演示论文的骨架。全部为虚构内容。
PAPERS = [
    {
        "arxiv": "2401.00001",
        "title": "Sparse Mixture-of-Experts Routing with Learned Capacity Control",
        "authors": "Wei Zhang, Elena Petrova, Kenji Nakamura",
        "abstract": (
            "We present a routing mechanism for sparse mixture-of-experts layers that "
            "learns its own token capacity instead of relying on a fixed assignment. "
            "The method reduces dropped tokens by 43% at equal compute, and we show "
            "on three translation benchmarks that the learned capacity correlates "
            "with token difficulty rather than sequence position."
        ),
        "sections": [
            ("1 Introduction", [
                "Sparse mixture-of-experts (MoE) layers scale model capacity without a "
                "proportional increase in compute, because each token is routed to only a "
                "small subset of experts.",
                "Prior work fixes the number of tokens each expert receives. This choice is "
                "made before training and is never revisited, which forces a compromise "
                "between dropping tokens and wasting capacity.",
                "我们提出的方法让容量本身成为可学习的参数。",
            ]),
            ("2 Method", [
                "Let x denote the token representation and let E = {e_1, ..., e_n} be the "
                "expert parameters. A router produces logits, and the top-k experts are "
                "selected by a softmax over those logits.",
                "The capacity c is parameterized as a smooth function of the routing "
                "entropy, so that experts facing difficult tokens are allowed more slots.",
                "自注意力机制在这里并不适用，因为路由决策是逐 token 独立的。",
            ]),
            ("3 Experiments", [
                "We evaluate on WMT14 En-De, WMT17 Zh-En, and a held-out internal set. "
                "All models are trained with the same compute budget.",
                "Our method reaches 28.4 BLEU at 1.2x the training cost of the dense "
                "baseline, compared with 27.9 BLEU for the fixed-capacity variant.",
                "图 3 显示了容量分布随训练推进的变化。",
            ]),
            ("4 Limitations", [
                "The entropy proxy is a heuristic and may not transfer to modalities "
                "beyond text. We did not evaluate on vision or speech.",
            ]),
        ],
        "references": [
            "Shazeer et al. Outrageously Large Neural Networks. ICLR 2017.",
            "Fedus et al. Switch Transformers. JMLR 2022.",
            "Lewis et al. BASE Layers. ICML 2021.",
        ],
    },
    {
        "arxiv": "2402.00002",
        "title": "A Benchmark for Long-Context Retrieval in Scientific Literature",
        "authors": "Marta Silva, Ahmed Hassan, Li Chen",
        "abstract": (
            "Existing retrieval benchmarks use short passages and do not reflect the "
            "structure of scientific papers. We introduce a benchmark of 4,200 queries "
            "over 12,000 papers, with citations verified against the source text. "
            "Strong dense retrievers lose 31 points of recall moving from paragraph-level "
            "to paper-level retrieval."
        ),
        "sections": [
            ("1 Introduction", [
                "Retrieval evaluation has largely converged on short-passage benchmarks.",
                "Scientific papers differ in ways that matter: they are long, heavily "
                "structured, and contain claims that depend on other claims.",
                "我们的基准测试覆盖了跨章节的推理问答。",
            ]),
            ("2 Dataset Construction", [
                "We sample papers from three arXiv categories and annotate queries with "
                "verified page-level citations.",
                "Each query is answerable from at most three pages, but those pages are "
                "not necessarily contiguous.",
            ]),
            ("3 Results", [
                "BM25 remains competitive on exact terminology queries, while dense "
                "retrievers lead on paraphrased questions.",
                "Hybrid retrieval with reciprocal rank fusion outperforms both by 4.1 "
                "points on the combined metric.",
                "知识蒸馏的方法在这里没有带来提升。",
            ]),
        ],
        "references": [
            "Thakur et al. BEIR. NeurIPS 2021.",
            "Muennighoff et al. MTEB. 2023.",
        ],
    },
    {
        "arxiv": "2403.00003",
        "title": "On the Difficulty of Reproducing Attention Visualizations",
        "authors": "Jonas Berg, Priya Raman",
        "abstract": (
            "Attention weights are widely used as explanations, but we find that small "
            "implementation differences change the resulting visualizations substantially. "
            "Across 18 public repositories we observe four distinct conventions for "
            "averaging attention heads, and these conventions disagree on which tokens "
            "appear salient in 27% of cases."
        ),
        "sections": [
            ("1 Introduction", [
                "Attention visualization is a common way to argue that a model has "
                "learned a linguistically meaningful pattern.",
                "We ask whether such visualizations are reproducible across codebases.",
            ]),
            ("2 Findings", [
                "We identify four conventions: averaging before softmax, averaging after "
                "softmax, taking the maximum over heads, and selecting a single head.",
                "在 27% 的样本中，不同约定指出不同的显著词。",
            ]),
            ("3 Recommendations", [
                "Report the aggregation convention explicitly, and publish the raw "
                "attention tensors alongside any visualization.",
            ]),
        ],
        "references": [
            "Jain & Wallace. Attention is not Explanation. NAACL 2019.",
            "Wiegreffe & Pinter. Attention is not not Explanation. EMNLP 2019.",
        ],
    },
]


def make_pdf(spec: dict, out_path: Path, *, corrupt_title: bool = False) -> None:
    """生成一份看起来像模像样的论文 PDF。"""
    doc = pymupdf.open()

    meta = {
        "title": spec["title"],
        "author": spec["authors"],
        "subject": f"arXiv:{spec['arxiv']}",
        "keywords": "machine learning, demo",
        "creator": "kb seed script",
    }
    if corrupt_title:
        # 模拟 Word 导出的 PDF：元数据里的标题是垃圾
        meta["title"] = os.path.basename(out_path)
    doc.set_metadata(meta)

    def new_page():
        page = doc.new_page(width=595, height=842)  # A4
        return page

    # ---- 首页 ----
    page = new_page()
    y = 90.0
    page.insert_text((72, y), spec["title"], fontsize=17, fontname="hebo")
    y += 30
    page.insert_text((72, y), spec["authors"], fontsize=11)
    y += 18
    page.insert_text((72, y), "arXiv:" + spec["arxiv"], fontsize=9, color=(0.4, 0.4, 0.4))
    y += 34

    page.insert_text((72, y), "Abstract", fontsize=13, fontname="hebo")
    y += 20
    y = write_paragraph(page, spec["abstract"], y, width=88)

    # 首页接摘要后面继续写第 1 节。
    # 注意要把第 1 节**全部**段落写完——只写字号最大的那段，
    # 后面的段落就会凭空消失，而这在生成阶段毫无提示。
    y += 18
    first_title, first_paras = spec["sections"][0]
    page.insert_text((72, y), first_title, fontsize=13, fontname="hebo")
    y += 20
    for para in first_paras:
        if y > 720:
            page = new_page()
            y = 90
        y = write_paragraph(page, para, y, width=88)
        y += 10

    # ---- 其余章节（第 1 节已经在上面写完，这里从第 2 节开始）----
    for title, paragraphs in spec["sections"][1:]:
        page = new_page()
        y = 90.0
        page.insert_text((72, y), title, fontsize=13, fontname="hebo")
        y += 22
        for para in paragraphs:
            if y > 700:
                page = new_page()
                y = 90
            y = write_paragraph(page, para, y, width=88)
            y += 10

    # ---- 参考文献 ----
    page = new_page()
    y = 90.0
    page.insert_text((72, y), "References", fontsize=13, fontname="hebo")
    y += 22
    for i, ref in enumerate(spec["references"], 1):
        y = write_paragraph(page, f"[{i}] {ref}", y, width=88, fontsize=9.5)
        y += 6

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    doc.close()


def write_paragraph(page, text: str, y: float, *, width: int, fontsize: float = 10.5) -> float:
    """把一段文字按宽度折行写进页面，返回新的 y。

    注意：这里用的是等宽近似（每行 width 个字符）。真正的中文混排需要
    字体度量，但对演示数据来说足够了——而且插入 CJK 字符时需要显式指定
    支持中文的字体，否则 PyMuPDF 会画出空白。
    """
    import textwrap

    # PyMuPDF 内置字体不含 CJK 字形，纯 ASCII 折行后再单独处理中文
    lines = textwrap.wrap(text, width=width) or [""]
    for line in lines:
        if y > 790:
            break
        if any(ord(ch) > 0x2E80 for ch in line):
            # 含中文的行用内置 CJK 字体
            try:
                page.insert_text((72, y), line, fontsize=fontsize, fontname="china-s")
            except Exception:
                page.insert_text((72, y), line.encode("ascii", "ignore").decode() or "…",
                                 fontsize=fontsize)
        else:
            page.insert_text((72, y), line, fontsize=fontsize)
        y += fontsize * 1.45
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="生成演示用假论文")
    parser.add_argument("target", type=Path, help="输出目录")
    parser.add_argument("--duplicates", action="store_true",
                        help="额外生成重复文件与近似标题，用于测试去重")
    parser.add_argument("--clean", action="store_true", help="先清空目标目录")
    args = parser.parse_args()

    target: Path = args.target
    if args.clean and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    # 分成子目录，验证递归扫描
    layout = ["2024/attention", "2024/retrieval", "2023/misc", ""]

    written = []
    for index, spec in enumerate(PAPERS):
        subdir = target / layout[index % len(layout)]
        safe_title = "".join(
            ch if ch.isalnum() or ch in " -_" else "_" for ch in spec["title"]
        )[:70].strip()
        path = subdir / f"{spec['arxiv'].replace('.', '_')}_{safe_title}.pdf"
        make_pdf(spec, path, corrupt_title=(index == 2))
        written.append(path)

    if args.duplicates:
        # 1) 内容完全相同的副本（不同路径）
        dup_dir = target / "duplicates"
        dup_dir.mkdir(exist_ok=True)
        shutil.copy2(written[0], dup_dir / "copy-of-first.pdf")

        # 2) 近似标题：换大小写、加标点、改副标题
        spec = dict(PAPERS[1])
        spec["title"] = "A Benchmark for Long Context Retrieval in Scientific Literature!"
        spec["arxiv"] = "2402.09999"  # 不同 arXiv 号，只能靠标题相似度发现
        make_pdf(spec, dup_dir / "near-duplicate-title.pdf")

        # 3) 同一篇论文的两个版本（arXiv v1 / v2 的常见情形）
        spec2 = dict(PAPERS[0])
        spec2["arxiv"] = "2401.00001v2"
        make_pdf(spec2, target / "2024/attention" / "v2-preprint.pdf")

    print(f"已在 {target} 生成 {len(written)} 篇演示论文" + ("（含重复样本）" if args.duplicates else ""))
    for path in sorted(target.rglob("*.pdf")):
        size = path.stat().st_size
        print(f"  {path.relative_to(target)}  ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
