"""运行时设置：声明式 schema + 数据库存储 + 密钥加密。

这里管的是「数据库打开之后才能读到」的那一层配置：论文目录在哪、用哪个模型、
检索取几条、界面什么主题。它们存在 ``settings`` 表里，网页端可改。

设计要点：

**schema 与存储分离。** 本模块声明每个配置项的类型、默认值、取值范围、
展示分组；数据库只负责存值。加一个配置项只需要在这里加一行，
不需要写迁移、也不需要改前端表单（表单由 schema 渲染出来）。

**每个值都能说清自己从哪来。** ``get_with_source()`` 返回 ``(值, 来源)``，
来源是 ``db``（用户改过）或 ``default``（用的默认值）。设置页据此显示
「已自定义 / 默认」标记——没有这个信息，用户看到一屏配置项根本不知道
哪些是自己改过的。

**密钥加密存储。** API Key 用 Fernet 加密后入库，读取接口一律返回掩码。
加密密钥来自启动级配置（``KB_SECRET_KEY`` 或数据目录下的 ``secret.key``）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)

SettingType = Literal["str", "int", "float", "bool", "list", "secret", "choice", "raw"]


@dataclass(frozen=True)
class SettingDef:
    """一个配置项的声明。"""

    key: str
    type: SettingType
    default: Any
    label: str
    group: str
    help: str = ""
    choices: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    # 高级项默认折叠，避免设置页一上来就糊满参数
    advanced: bool = False
    # 依赖其它配置项才有效时，给出那个键名（用于界面上的联动禁用）
    depends_on: str | None = None

    def coerce(self, raw: Any) -> Any:
        """把外部输入（表单字符串、JSON）转成正确类型并校验。

        表单传过来的永远是字符串，而配置项有类型。转换失败必须报错而不是
        静默用默认值——用户改了配置却不知道没生效，是最糟糕的失败方式。
        """
        if raw is None:
            return self.default

        if self.type == "bool":
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off", ""}:
                return False
            raise ValueError(f"{self.label} 需要是布尔值")

        if self.type == "int":
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.label} 需要是整数") from exc
            return self._check_range(value)

        if self.type == "float":
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.label} 需要是数字") from exc
            return self._check_range(value)

        if self.type == "list":
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, str):
                # 允许换行或逗号分隔的文本输入（路径列表就是这么填的）
                items = [p.strip() for p in raw.replace(",", "\n").splitlines()]
            else:
                raise ValueError(f"{self.label} 需要是列表")
            return [str(i).strip() for i in items if str(i).strip()]

        if self.type == "choice":
            value = str(raw)
            if self.choices and value not in self.choices:
                raise ValueError(f"{self.label} 只能是：{', '.join(self.choices)}")
            return value

        if self.type == "raw":
            # 复杂结构（dict/list/None），由程序写入，不做用户输入校验。
            # 设置页里以 JSON 只读展示。
            return raw

        # str / secret
        return str(raw)

    def _check_range(self, value: float) -> float:
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label} 不能小于 {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.label} 不能大于 {self.maximum}")
        return value

    def to_public(self) -> dict:
        """给前端渲染表单用的元数据。"""
        return {
            "key": self.key,
            "type": self.type,
            "default": "" if self.type == "secret" else self.default,
            "label": self.label,
            "group": self.group,
            "help": self.help,
            "choices": list(self.choices) if self.choices else None,
            "min": self.minimum,
            "max": self.maximum,
            "advanced": self.advanced,
            "depends_on": self.depends_on,
        }


# --------------------------------------------------------------------------
# 配置项声明
# --------------------------------------------------------------------------

GROUPS = {
    "paths": "路径",
    "scan": "扫描与去重",
    "ingest": "正文解析",
    "llm": "模型",
    "embedding": "向量嵌入",
    "retrieval": "检索",
    "websearch": "联网搜索",
    "notes": "笔记与同步",
    "appearance": "外观主题",
    "jobs": "后台任务",
    "security": "安全",
}

_DEFS: list[SettingDef] = [
    # ---------------- 路径 ----------------
    SettingDef(
        "paths.papers_roots", "list", ["/mnt/papers"], "论文根目录", "paths",
        "递归扫描这些目录下的 PDF。可以填多个。",
    ),
    SettingDef(
        "paths.notes_root", "str", "/mnt/kb-notes", "笔记目录", "paths",
        "笔记会以 Markdown 文件形式写到这里，可直接用 Obsidian 打开。",
    ),
    SettingDef(
        "paths.codes_root", "str", "/mnt/kb-codes", "代码仓库目录", "paths",
        "开源代码仓库的存放位置，用于与论文建立关联。",
    ),
    SettingDef(
        "paths.auto_create", "bool", True, "自动创建缺失的目录", "paths",
        "开启后，保存路径时若目录不存在会自动创建。",
    ),

    # ---------------- 扫描与去重 ----------------
    SettingDef(
        "scan.max_depth", "int", 12, "最大递归深度", "scan",
        "防止误把根目录设成 / 之类的路径后无限扫描。",
        minimum=1, maximum=64, advanced=True,
    ),
    SettingDef(
        "scan.follow_symlinks", "bool", False, "跟随符号链接", "scan",
        "默认关闭：符号链接容易造成循环扫描和越出根目录的读取。",
        advanced=True,
    ),
    SettingDef(
        "scan.ignore_hidden", "bool", True, "忽略隐藏文件与目录", "scan", advanced=True,
    ),
    SettingDef(
        "scan.exclude_globs", "list", [".*", "node_modules", "__pycache__", "*.tmp"],
        "排除的目录/文件名", "scan",
        "支持 shell 通配符。每行一个。",
    ),
    SettingDef(
        "scan.dedup_exact", "bool", True, "内容完全相同视为重复", "scan",
        "对文件内容做哈希。只对新文件和变更过的文件计算。",
    ),
    SettingDef(
        "scan.dedup_identifier", "bool", True, "DOI / arXiv 号相同视为重复", "scan",
    ),
    SettingDef(
        "scan.dedup_fuzzy", "bool", True, "标题相似时提示可能的重复", "scan",
        "相似的结果不会自动合并，而是列出来等你在页面上确认。",
    ),
    SettingDef(
        "scan.fuzzy_threshold", "float", 92.0, "标题相似度阈值", "scan",
        "0-100。调低会提示更多候选（也更容易误报）。",
        minimum=50.0, maximum=100.0, advanced=True,
        depends_on="scan.dedup_fuzzy",
    ),

    # ---------------- 正文来源 ----------------
    SettingDef(
        "ingest.prefer_latex", "bool", True, "优先使用 LaTeX 源码", "ingest",
        "论文有 LaTeX 源码时优先解析源码而不是 PDF。源码里章节、公式、"
        "图表标题、引用键都是显式声明的，比从 PDF 反推结构准确得多；"
        "页码则用 PDF 反查章节标题得到。",
    ),
    SettingDef(
        "ingest.fetch_arxiv_source", "bool", True, "自动从 arXiv 下载源码", "ingest",
        "对带 arXiv 编号的论文，自动下载其源码包（会访问外网，"
        "并遵守 arXiv 的抓取频率限制）。关闭后只用本地已有的源码。",
        depends_on="ingest.prefer_latex",
    ),
    SettingDef(
        "ingest.pdf_engine", "choice", "auto", "PDF 解析方式", "ingest",
        "markdown 用 pymupdf4llm 先把 PDF 转成 Markdown 再解析，"
        "标题层级识别明显更准（能还原 3.2.1 这样的编号层级）；"
        "heuristic 用字号启发式，速度快但结构粗。auto 优先用前者、失败时退回。",
        choices=("auto", "markdown", "heuristic"),
    ),

    # ---------------- 模型 ----------------
    SettingDef(
        "llm.provider", "choice", "anthropic", "服务商", "llm",
        "anthropic 用官方 SDK，支持 PDF 原生解析与页码级引用；"
        "openai_compatible 用于 DeepSeek / Qwen / vLLM / Ollama 等。",
        choices=("anthropic", "openai_compatible"),
    ),
    SettingDef(
        "llm.api_key", "secret", "", "API Key", "llm",
        "仅在保存时提交，读取时只返回掩码。",
    ),
    SettingDef(
        "llm.base_url", "str", "", "API 地址", "llm",
        "留空用 Anthropic 官方端点；填第三方兼容网关的地址，"
        "例如 https://api.deepseek.com/anthropic。",
    ),
    SettingDef(
        "llm.auth_mode", "choice", "auto", "认证方式", "llm",
        "auto 会按地址判断：官方端点用 x-api-key，其它用 Authorization: Bearer。"
        "第三方网关多半需要 Bearer。",
        choices=("auto", "api_key", "bearer"), advanced=True,
    ),
    SettingDef(
        "llm.deep_model", "str", "claude-opus-5-5", "深度阅读模型", "llm",
        "负责精读论文、生成笔记、多轮问答。",
    ),
    SettingDef(
        "llm.fast_model", "str", "claude-haiku-4-5", "轻量任务模型", "llm",
        "负责打标签、分类等高频低成本调用。",
    ),
    SettingDef(
        "llm.max_tokens", "int", 16000, "单次最大输出 token", "llm",
        "注意：思考内容与正文**共用**这个额度。实测推理型模型回答一个是非题"
        "也会消耗两千多个字符的思考，额度给小了会导致正文一个字都出不来。",
        minimum=256, maximum=128000,
    ),
    SettingDef(
        "llm.effort", "choice", "medium", "思考深度", "llm",
        "Opus 5.5 的默认档位是 medium；难题可调到 high 或 xhigh，"
        "代价是更慢更贵。部分兼容端点会忽略此项。",
        choices=("low", "medium", "high", "xhigh", "max"), advanced=True,
    ),
    # 由能力探测写入，不由用户直接编辑。放在设置里是为了「探测一次、长期生效」，
    # 不必每次构造 Provider 都重新探测（那要花钱）。
    SettingDef(
        "llm.capabilities", "raw", None, "已探测的能力", "llm",
        "由「能力实测」写入。决定结构化抽取走哪条路径、是否依赖提示缓存等。",
        advanced=True,
    ),

    # ---------------- 用量与花费 ----------------
    # 默认**开着**一道 token 闸（500 万 / 24 小时）。一次全库重读实测约 132 万
    # token，所以正常使用碰不到它，但一个失控的循环会在几十分钟内撞上来。
    # 闸门超限时抛异常、不降级——静默降级会让「结果变差」看起来像「模型不行」。
    SettingDef(
        "llm.budget_tokens", "int", 5_000_000, "用量上限（token）", "llm",
        "滚动 24 小时内累计 token 上限，超过就拒绝调用并报错。"
        "设为 0 关闭。一次全库重读约 132 万 token，可据此估算。"
        "这道闸不需要填单价就能生效。",
        minimum=0,
    ),
    SettingDef(
        "llm.budget_usd", "float", 0.0, "花费上限（美元）", "llm",
        "滚动 24 小时内的估算花费上限，设为 0 关闭。"
        "**需要先填下面两个单价才会生效**——价格会变，我们不替你猜。",
        minimum=0.0,
    ),
    SettingDef(
        "llm.price_input_per_mtok", "float", 0.0, "输入单价（美元/百万 token）", "llm",
        "只用于把 token 折算成金额显示，以及上面那道花费闸。留 0 表示不计价。",
        minimum=0.0, advanced=True,
    ),
    SettingDef(
        "llm.price_output_per_mtok", "float", 0.0, "输出单价（美元/百万 token）", "llm",
        "输出通常比输入贵数倍，所以分开填。留 0 表示不计价。",
        minimum=0.0, advanced=True,
    ),

    # ---------------- 嵌入 ----------------
    SettingDef(
        "embedding.provider", "choice", "local", "嵌入服务", "embedding",
        "local 用 fastembed 在本机跑 ONNX 模型——不需要 API Key，模型下载后可离线，"
        "中文提问命中英文论文主要靠它。OpenAI 兼容接口则走远程。"
        "（注意 Anthropic 与 DeepSeek 都不提供嵌入接口。）",
        choices=("local", "openai_compatible"),
    ),
    SettingDef(
        "embedding.model", "str", "BAAI/bge-small-zh-v1.5", "嵌入模型", "embedding",
        "本地模型留空就不启用向量检索。中文语料推荐 bge-small-zh-v1.5（约 100MB）；"
        "中英混排想要更好效果可以换 BAAI/bge-m3（约 2GB）。",
    ),
    SettingDef(
        "embedding.base_url", "str", "", "嵌入 API 地址", "embedding",
        depends_on="embedding.provider",
    ),
    SettingDef(
        "embedding.api_key", "secret", "", "嵌入 API Key", "embedding",
        depends_on="embedding.provider",
    ),
    SettingDef(
        "embedding.dim", "int", 0, "向量维度", "embedding",
        "0 表示首次调用时自动探测。填错会导致向量表不匹配。",
        minimum=0, maximum=8192,
    ),
    SettingDef(
        "embedding.batch_size", "int", 32, "批大小", "embedding",
        minimum=1, maximum=512, advanced=True,
    ),

    # ---------------- 检索 ----------------
    SettingDef(
        "retrieval.top_k", "int", 8, "返回条数", "retrieval",
        minimum=1, maximum=100,
    ),
    SettingDef(
        "retrieval.use_vector", "bool", True, "启用向量检索", "retrieval",
        "关闭后只用全文检索（BM25）。没有配置嵌入模型时自动关闭。",
    ),
    SettingDef(
        "retrieval.rrf_k", "int", 60, "RRF 融合参数", "retrieval",
        "倒数排名融合的平滑常数，一般不用改。",
        minimum=1, maximum=1000, advanced=True,
    ),
    SettingDef(
        "retrieval.query_expansion", "bool", True, "查询词扩展", "retrieval",
        "中文提问时，先让模型把它翻成英文技术词再检索。语料多为英文论文时"
        "这一步提升明显——全文检索是字面匹配，中文查询根本够不着英文正文。"
        "每次唯一的查询会多一次小模型调用（结果按查询缓存）。",
    ),
    SettingDef(
        "retrieval.group_by_paper", "bool", True, "按论文聚合结果", "retrieval",
        "同一篇论文命中多段时合并显示，避免一篇论文刷屏。",
    ),
    SettingDef(
        "chunk.size", "int", 800, "分块大小（token）", "retrieval",
        minimum=128, maximum=4096, advanced=True,
    ),
    SettingDef(
        "chunk.overlap", "int", 120, "分块重叠（token）", "retrieval",
        minimum=0, maximum=1024, advanced=True,
    ),

    # ---------------- 联网搜索 ----------------
    SettingDef(
        "websearch.enabled", "bool", True, "允许模型联网搜索", "websearch",
        "知识库检索不到时，模型可以自己决定要不要上网查。"
        "回答里会区分标出哪些内容来自联网、并给出可点击的来源链接。",
    ),
    SettingDef(
        "websearch.academic", "bool", True, "检索学术文献", "websearch",
        "接 OpenAlex / Crossref / arXiv / Semantic Scholar，查某方向的论文、"
        "引用关系与正式发表版本。**不需要 API Key。**"
        "几个源会互相印证：同一篇工作被多个源返回时排序会提前。",
    ),
    SettingDef(
        "websearch.code", "bool", True, "检索代码仓库", "websearch",
        "接 GitHub 搜索，查某个方法在别的仓库里怎么实现的。不需要 Key。",
    ),
    SettingDef(
        "websearch.discussions", "bool", True, "检索社区讨论", "websearch",
        "接 HackerNews。**不需要 Key。**"
        "论文告诉你作者声称什么，讨论区告诉你同行信不信、有没有人复现失败——"
        "这是学术接口给不了的信息。",
    ),
    SettingDef(
        "websearch.tavily_api_key", "secret", "", "Tavily API Key", "websearch",
        "通用网页搜索（官方文档、技术博客、issue 讨论）需要它。"
        "**留空则跳过通用网页检索**，学术、代码与讨论检索不受影响。"
        "（实测这个网络环境下，免 Key 的通用搜索都不可用："
        "DuckDuckGo / Brave / SearXNG 的 TLS 握手超时，Mojeek 返回验证码页。）",
    ),
    SettingDef(
        "websearch.max_results", "int", 6, "每次返回条数", "websearch",
        minimum=1, maximum=20,
    ),
    SettingDef(
        "websearch.fetch_pages", "bool", True, "抓取网页正文", "websearch",
        "只用搜索摘要的话信息量常常不够；开启后会抓取正文并抽取主要内容。"
        "代价是每次联网慢几秒。",
    ),
    SettingDef(
        "websearch.max_page_chars", "int", 4000, "单页正文上限（字符）", "websearch",
        minimum=500, maximum=20000, advanced=True,
    ),
    SettingDef(
        "websearch.cache_ttl_minutes", "int", 60, "检索结果缓存（分钟）", "websearch",
        "同一查询在这么多分钟内直接复用上次结果，不再联网。"
        "联网结果几分钟内不会变，而抓正文是整个链路最慢的一步；"
        "几个源本身也会抖（实测 OpenAlex 会间歇性超时），缓存让它们在抖的时候仍有结果。"
        "**设为 0 关闭。**",
        minimum=0, maximum=10080,
    ),

    # ---------------- 笔记 ----------------
    SettingDef(
        "notes.write_to_disk", "bool", True, "笔记同步到磁盘", "notes",
        "关闭后笔记只存在于数据库里。开启后才能用 Obsidian 等外部工具编辑。",
    ),
    SettingDef(
        "notes.write_mode", "choice", "app", "冲突处理方式", "notes",
        "app：本应用写入前检测外部改动，冲突时保留两份让你选；"
        "external：本应用不覆盖已存在的文件，只追加到冲突目录。",
        choices=("app", "external"),
        depends_on="notes.write_to_disk",
    ),
    SettingDef(
        "notes.dir_template", "str", "{year}/{paper_slug}", "笔记子目录结构", "notes",
        "可用变量：{year} {paper_slug} {paper_title} {kind}。",
        depends_on="notes.write_to_disk",
    ),
    SettingDef(
        "notes.line_ending", "choice", "lf", "换行符", "notes",
        "lf 在各平台都正常；crlf 仅为兼容个别 Windows 工具。",
        choices=("lf", "crlf"), advanced=True,
    ),

    # ---------------- 外观 ----------------
    SettingDef(
        "appearance.theme", "choice", "system", "主题", "appearance",
        "system 跟随操作系统。",
        choices=("system", "light", "dark"),
    ),
    SettingDef(
        "appearance.accent", "choice", "indigo", "主题色", "appearance",
        choices=("indigo", "blue", "teal", "green", "amber", "rose", "violet"),
    ),
    SettingDef(
        "appearance.font_size", "choice", "medium", "字号", "appearance",
        choices=("small", "medium", "large"),
    ),
    SettingDef(
        "appearance.density", "choice", "comfortable", "界面密度", "appearance",
        choices=("compact", "comfortable", "spacious"),
    ),
    SettingDef(
        "appearance.radius", "choice", "medium", "圆角", "appearance",
        choices=("none", "small", "medium", "large"), advanced=True,
    ),
    SettingDef(
        "appearance.reader_font", "choice", "serif", "阅读器正文字体", "appearance",
        choices=("serif", "sans"), advanced=True,
    ),

    # ---------------- 任务 ----------------
    SettingDef(
        "jobs.concurrency", "int", 2, "并发任务数", "jobs",
        "同时执行的后台任务数量。解析 PDF 与调用模型都是重 IO，"
        "调太高反而会互相拖慢。",
        minimum=1, maximum=8,
    ),
    SettingDef(
        "jobs.lease_seconds", "int", 300, "任务租约（秒）", "jobs",
        "超过这个时间没有心跳的任务，会被认为已崩溃并重新排队。",
        minimum=60, maximum=3600, advanced=True,
    ),
    SettingDef(
        "jobs.keep_events_days", "int", 14, "任务日志保留天数", "jobs",
        minimum=1, maximum=365, advanced=True,
    ),

    # ---------------- 安全 ----------------
    SettingDef(
        "security.rate_limit", "str", "240/minute", "接口限流", "security",
        "形如 120/minute、10/second。留空表示不限流。",
    ),
    SettingDef(
        "security.audit_enabled", "bool", True, "记录接口调用审计", "security",
    ),
]

BY_KEY: dict[str, SettingDef] = {d.key: d for d in _DEFS}


def definitions() -> list[SettingDef]:
    return list(_DEFS)


def grouped_definitions() -> dict[str, list[SettingDef]]:
    out: dict[str, list[SettingDef]] = {g: [] for g in GROUPS}
    for d in _DEFS:
        out.setdefault(d.group, []).append(d)
    return out


# --------------------------------------------------------------------------
# 设置服务
# --------------------------------------------------------------------------


class SettingsError(ValueError):
    """设置项校验失败。消息面向用户，会直接显示在设置页上。"""


class Settings:
    """运行时设置的读写门面。

    带进程内缓存：设置项在每次请求里会被读很多次（渲染模板、服务层判断分支），
    而它们极少变化。写入时主动失效，不存在读到旧值的问题。
    """

    def __init__(self, app: Any, fernet: Fernet | None = None):
        self._app = app
        self._fernet = fernet
        self._cache: dict[str, Any] | None = None

    # --- 上下文 ---
    def _session(self):
        """取得一个数据库会话。

        自己推 app context 而不是要求调用方准备好：设置会在三种截然不同的
        环境里被读取——Web 请求、后台 worker 线程、CLI 命令，其中后两者
        本来就没有请求上下文。让每处调用点记得 ``with app.app_context()``
        是把一个必然会被忘记的约定散播到全代码库。
        """
        from .extensions import db

        return db.session

    def _ctx(self):
        return self._app.app_context()

    # --- 缓存 ---
    def invalidate(self) -> None:
        self._cache = None

    def _load(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        from .models import Setting

        with self._ctx():
            rows = self._session().query(Setting).all()
            self._cache = {r.key: r.value for r in rows}
        return self._cache

    # --- 读 ---
    def get(self, key: str, default: Any = None) -> Any:
        definition = BY_KEY.get(key)
        raw = self._load().get(key)

        if raw is None:
            if definition is not None:
                return definition.default
            return default

        if definition is not None:
            try:
                return definition.coerce(raw)
            except ValueError:
                # 数据库里的值不合法（手工改过、旧版本残留）——
                # 退回默认值而不是让整个页面炸掉，但要留下日志。
                log.warning("设置项 %s 的值不合法（%r），已回退到默认值", key, raw)
                return definition.default
        return raw

    def get_secret(self, key: str) -> str:
        """解密读取密钥。只在真正要调用模型时使用，绝不返回给前端。"""
        raw = self._load().get(key)
        if not raw:
            return ""
        if self._fernet is None:
            log.warning("没有配置加密密钥，无法解密 %s", key)
            return ""
        try:
            return self._fernet.decrypt(raw.encode()).decode()
        except (InvalidToken, AttributeError):
            log.warning("设置项 %s 解密失败——通常是 secret.key 被替换过，需要重新填写", key)
            return ""

    def has_secret(self, key: str) -> bool:
        return bool(self._load().get(key))

    @staticmethod
    def mask(value: str) -> str:
        """把密钥变成可用于展示的掩码。

        保留头尾是为了让用户能分辨「这是我那把 key 吗」——
        只显示 ``****`` 的话，有多把 key 时根本分不清。
        """
        if not value:
            return ""
        if len(value) <= 12:
            return "•" * len(value)
        return f"{value[:7]}…{value[-4:]}"

    def get_with_source(self, key: str) -> tuple[Any, str]:
        """返回 (值, 来源)。来源为 ``db`` 或 ``default``，供设置页显示标记。"""
        has_row = key in self._load()
        return self.get(key), ("db" if has_row else "default")

    def all(self, *, include_secrets: bool = False) -> dict[str, Any]:
        """所有设置项的值。``include_secrets=False`` 时密钥返回掩码。"""
        out: dict[str, Any] = {}
        for d in _DEFS:
            if d.type == "secret":
                raw = self.get_secret(d.key)
                out[d.key] = raw if include_secrets else self.mask(raw)
            else:
                out[d.key] = self.get(d.key)
        return out

    def all_with_source(self) -> dict[str, dict]:
        """给设置页用的完整结构：值 + 来源 + 元数据。"""
        data: dict[str, dict] = {}
        stored = self._load()
        for d in _DEFS:
            if d.type == "secret":
                value = self.mask(self.get_secret(d.key))
                has_value = self.has_secret(d.key)
            else:
                value = self.get(d.key)
                has_value = d.key in stored
            data[d.key] = {
                "value": value,
                "source": "db" if d.key in stored else "default",
                "has_value": has_value,
                **d.to_public(),
            }
        return data

    # --- 写 ---
    def set(self, key: str, raw: Any, *, updated_by: str | None = None) -> Any:
        definition = BY_KEY.get(key)
        if definition is None:
            raise SettingsError(f"未知的配置项：{key}")

        if definition.type == "secret":
            return self._set_secret(key, raw, updated_by=updated_by)

        try:
            value = definition.coerce(raw)
        except ValueError as exc:
            raise SettingsError(str(exc)) from exc

        self._write(key, value, updated_by=updated_by)
        return value

    def _set_secret(self, key: str, raw: Any, *, updated_by: str | None) -> str:
        if self._fernet is None:
            raise SettingsError(
                "没有可用的加密密钥，无法安全保存 API Key。"
                "请先在 .env 里设置 KB_SECRET_KEY，或确保数据目录可写以自动生成。"
            )
        text = "" if raw is None else str(raw).strip()

        # 空字符串表示「清除」，掩码值表示「没改」，两者要区分开：
        # 设置页每次都会把整张表单提交上来，如果不区分，
        # 用户只改主题也会把 API Key 清成掩码字符串。
        if text == "":
            self._write(key, None, updated_by=updated_by)
            return ""

        # 掩码里的省略号与圆点不会出现在任何真实密钥中，据此识别「未修改」
        if "…" in text or "•" in text:
            return self.mask(self.get_secret(key))

        self._write(key, self._fernet.encrypt(text.encode()).decode(), updated_by=updated_by)
        return self.mask(text)

    def update_many(self, values: dict[str, Any], *, updated_by: str | None = None) -> dict:
        """批量更新。任一项校验失败则整批不生效——避免出现改了一半的状态。"""
        prepared: list[tuple[str, Any]] = []
        for key, raw in values.items():
            definition = BY_KEY.get(key)
            if definition is None:
                raise SettingsError(f"未知的配置项：{key}")
            if definition.type == "secret":
                continue  # 密钥单独走 _set_secret
            try:
                prepared.append((key, definition.coerce(raw)))
            except ValueError as exc:
                raise SettingsError(str(exc)) from exc

        for key, value in prepared:
            self._write(key, value, updated_by=updated_by)

        for key, raw in values.items():
            if BY_KEY.get(key) and BY_KEY[key].type == "secret":
                self._set_secret(key, raw, updated_by=updated_by)

        return self.all_with_source()

    def _write(self, key: str, value: Any, *, updated_by: str | None) -> None:
        from .models import Setting

        with self._ctx():
            session = self._session()
            row = session.query(Setting).filter_by(key=key).one_or_none()
            if row is None:
                row = Setting(key=key, value=value, updated_by=updated_by)
                session.add(row)
            else:
                row.value = value
                row.updated_by = updated_by
            session.commit()
        self.invalidate()

    # --- 便捷读取（带业务含义） ---
    @property
    def papers_roots(self) -> list[str]:
        return [p for p in self.get("paths.papers_roots") or [] if p]

    @property
    def notes_root(self) -> str:
        return self.get("paths.notes_root") or ""

    @property
    def codes_root(self) -> str:
        return self.get("paths.codes_root") or ""

    @property
    def vector_enabled(self) -> bool:
        return bool(self.get("retrieval.use_vector")) and bool(self.get("embedding.model"))


def make_fernet(secret_key: bytes) -> Fernet | None:
    try:
        return Fernet(secret_key)
    except Exception:
        log.exception("无法用启动密钥构造 Fernet，密钥相关功能将不可用")
        return None


__all__ = [
    "BY_KEY",
    "GROUPS",
    "SettingDef",
    "Settings",
    "SettingsError",
    "definitions",
    "grouped_definitions",
    "make_fernet",
]
