"""深度阅读流水线。

把论文变成笔记的过程被拆成若干**可缓存、可单独重跑**的阶段：

    extract -> structure -> summarize -> tag -> graph -> publish

每一步的产物按「指纹」缓存（论文内容哈希 + 阶段名 + 提示词版本 + 模型 + 参数）。
指纹没变就直接复用——调试提示词时，改动只影响下游阶段，
不必把整条链从头跑一遍，这在反复调提示词时是数量级的差别。

阶段之间只通过产物传递数据，不共享内存状态。这样任何一步失败都可以
单独重跑，而不是「整篇重来」。
"""

from .pipeline import PipelineResult, run_pipeline

__all__ = ["PipelineResult", "run_pipeline"]
