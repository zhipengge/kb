# kb — 论文知识库

把散落在磁盘上的论文 PDF、配套开源代码、以及自己写的笔记统一管理起来，
用 AI 做深度阅读与问答，并对外暴露接口供其他 agent 调用。

**这个仓库只有代码，不含任何论文、笔记或代码仓库内容。** 你的数据始终留在你自己
配置的目录里（默认 `/mnt/papers`、`/mnt/kb-notes`、`/mnt/kb-codes`），
可以用 Obsidian、git、任何编辑器直接打开。

---

## 快速开始

```bash
# 1. 安装依赖（Python 3.12+，本项目在 3.14 上开发）
pipenv sync --dev

# 2. 看环境是否就绪
pipenv run flask --app wsgi kb selfcheck

# 3. 启动
pipenv run python run.py
# 打开 http://127.0.0.1:5000
```

前端库**随仓库自带**（`kb/web/static/vendor/`），不用 CDN，也不会在
`pipenv sync` 之后才出现——知识库常部署在没有外网的机器上，引 CDN 的页面
在那里会直接白屏，所以这些资源是直接提交进仓库的。

需要升级或补装时再跑 `pipenv run python scripts/vendor_assets.py`：
它会从多个镜像依次尝试（实测某些网络环境下 `cdn.jsdelivr.net` 不可达）。

首次启动会自动创建数据目录、生成加密密钥、建库并写入默认配置。
然后到 **设置 → 路径** 填上你的论文目录，回概览页点「扫描目录」。

### 签发对外接口的密钥

```bash
pipenv run flask --app wsgi kb key create --name "我的 agent" --scope read
```

明文只在创建时显示一次。之后带上它调用：

```bash
curl -H "Authorization: Bearer kb_xxxxxxxx_..." \
     http://127.0.0.1:5000/api/v1/system/health
```

---

## 接给外部 agent

> **如果你是要「使用」这个知识库的 agent，直接读
> [`docs/agent-guide.md`](docs/agent-guide.md)。**
> 那份文档面向调用方：怎么鉴权、核心的 `search → read → cite` 范式、
> 引用可信度三态、以及哪些调用要花钱。本节讲的是**怎么把它接起来**（给部署的人看）。

两种方式，**按客户端挑一种即可**，不用都接。

### 一、MCP（推荐）

任何支持 MCP 的客户端（Claude Code / Claude Desktop / 各家 agent 框架）都能直接连。
**不需要先把 Web 服务跑起来**——stdio 方式由客户端自己拉起子进程，直接读同一个数据库。

```bash
claude mcp add kb -- "$(pipenv --venv)/bin/python" "$PWD/scripts/mcp_server.py"
```

写成 JSON 配置交给别的客户端：

```json
{"mcpServers": {"kb": {
  "command": "/绝对路径/.venv/bin/python",
  "args": ["/绝对路径/scripts/mcp_server.py"]
}}}
```

也可以走 HTTP（客户端拉不起子进程时用）。必须先起 Web 服务，再指向：

```
POST http://127.0.0.1:5000/api/v1/mcp      # 请求体是 JSON-RPC 2.0
Authorization: Bearer kb_xxxxxxxx_...
```

**自检**：`curl http://127.0.0.1:5000/api/v1/mcp`（GET）会返回协议版本与支持列表。

#### 能用的工具（10 个，全部只读）

| 工具 | 用途 | 调用模型 |
|---|---|---|
| `search` | 混合检索，返回带 `locator` 的片段。**入口工具** | 仅查询扩展 |
| `read_paper` | 按小节/页读论文正文，不传参返回目录 | 否 |
| `get_note` | 取 AI 精读笔记（中文，已消化过的内容） | 否 |
| `list_papers` / `list_notes` | 按年份、会议、状态、标签翻列表 | 否 |
| `get_paper` / `get_code` | 论文元数据 / 关联的开源仓库 | 否 |
| `list_tags` / `stats` | 标签词表 / 知识库总览 | 否 |
| `ask` | 让模型基于检索结果作答，**带引用校验** | 是（端到端） |

**注意 `search` 不是完全免费的**：它默认会触发一次「查询扩展」的轻量模型调用
（把中文提问翻成英文术语），实测约 1.6 秒。同一查询词有缓存，重复查不再调用；
把 `retrieval.query_expansion` 关掉就变成纯本地检索。

典型顺序：`search` 找材料 → `read_paper` / `get_note` 读上下文 → 引用时带上 `locator`。
需要综合多篇直接成段回答时才用 `ask`。

**只提供读工具是有意的。** 写操作走下面的 REST 接口——那边有 `write` 作用域的 Key、
乐观锁、字段白名单，而且**由人决定要不要签发**。给模型一个删除工具，它总会在某次
误解指令时真的删掉。

工具清单只有一份定义（`kb/services/agent_tools.py`），MCP、下面的指南都从它生成，
不存在「文档和实现对不上」的可能。

#### 几个实现上的取舍

- **协议按 2026-07-28 版实现**（无状态：版本号在每条请求的 `_meta` 里，
  用 `server/discover` 自报能力），同时兼容 2025-11-25 及更早的 `initialize` 握手。
  现实里的客户端不会同时升级，只做一边就会有一半连不上。
- **手写协议层，不装官方 SDK**（`kb/mcp.py`）。官方 `mcp` 包要拖进 14 个新依赖，
  包括 starlette + uvicorn —— 本项目已经跑着 Flask，为一个 stdio 工具服务再架一套
  Web 框架不划算。真正要实现的协议面只有四个方法。
- **stdio 进程不启动内嵌 worker**。MCP 是按需查询的进程，可能同时开好几个，
  每个都拖一份任务循环去抢同一个任务队列只会互相打架。

### 二、REST 接口

需要脚本化调用、或者要写数据时用它。

```bash
curl -H "Authorization: Bearer kb_xxx" \
     "http://127.0.0.1:5000/api/v1/search?q=diffusion+policy&limit=5"
```

**先读这份再动手**，它比你自己翻源码快：

```bash
curl -H "Authorization: Bearer kb_xxx" http://127.0.0.1:5000/api/v1/agent-guide   # 纯文本，可直接喂给模型
curl http://127.0.0.1:5000/api/v1/openapi.json                                     # OpenAPI 3.1（无需鉴权）
```

`agent-guide` 是给模型读的自述文档（纯文本，省 token），`openapi.json` 的路径
**从真实路由自动生成**，不会出现「文档里有、实际 404」的情况。

---

## 正文解析：优先用 LaTeX 源码

对带 arXiv 编号的论文，系统会**自动下载并解析 LaTeX 源码**，而不是从 PDF 反推结构。

原因是 PDF 是排版结果，结构信息在排版时就被压掉了；而源码里这些是显式声明的：

| | LaTeX 源码 | PDF |
|---|---|---|
| 章节层级 | `\section{}` 直接给出 | 靠字号、缩进猜，双栏易错 |
| 公式 | 原始 LaTeX，可渲染 | 抽取成文本后符号大量丢失 |
| 引用关系 | `\cite{key}` 直接可得 | 只有「[1]」，要反解参考文献表 |
| 图注 | 与图一一对应 | 位置关系需要推测 |
| 页码 | 没有（要编译才知道） | 天然就有 |

**页码用 PDF 补。** 解析出章节后，拿章节标题去 PDF 里搜索定位，得到起始页。
这样既拿到源码的准确结构，又保住了引用跳转所需的页码。

拿不到源码时（作者只传了 PDF、非 arXiv 论文、网络不可用）完全退回 PDF 解析。

配置在「设置 → 正文解析」：

* **优先使用 LaTeX 源码** —— 关掉就一律用 PDF
* **自动从 arXiv 下载源码** —— 关掉则只用本地已有的源码包

下载的源码缓存在数据目录的 `cache/sources/` 下，重复索引不会重新下载。

### 两个实际处理过的坑

**被注释掉的草稿。** 论文源码里常留着大量注释掉的试验内容。实测
Attention Is All You Need 的源码有 **三分之一** 是注释（79891 → 53888 字符），
其中包含整段被注释掉的公式。不剥离的话，这些**论文里并不存在**的内容
会被建进索引，检索时会返回论文没说过的话。

**章节与正文错位。** 早期实现分两遍扫描：一遍切结构、一遍按相同规则切正文。
两遍的输入文本不同（一遍是 `\begin{document}` 环境内、一遍是全文），
偏移对不上，结果所有章节的正文都错位到了相邻章节——结论的文字挂在方法一节下面。
现在改为切分时就地取正文，不存在对齐问题。

---

## 图表与公式

公式以原始 LaTeX 保存并作为独立分块索引（`kind=formula`），
图注、表格标题同理（`kind=figure` / `kind=table`）。

这样检索时可以按类型过滤：「这个指标的数值是多少」只查表格，
「注意力的公式是什么」只查公式，不会被恰好包含它们的散文段落淹没。

前端的公式由 **KaTeX** 渲染。论文里的自定义宏（`\dmodel` 之类）KaTeX 不认，
遇到时会保留原文并标红，而不是让整页渲染中断。

> **引用的粒度取决于来源。** LaTeX 来源的引用精确到章节（`§3.2 Method`），
> PDF 来源的精确到页（`p.3`）。LaTeX 路径下页码是反查得到的，所以多数章节
> 两者都有。

---

## 设计要点

**磁盘是内容的载体，数据库是索引。** 笔记以 Markdown 文件形式落在
`/mnt/kb-notes`，带 YAML frontmatter，可直接用 Obsidian 编辑；数据库存元数据
与检索索引。外部改动会被检测到，冲突时**不自动覆盖任何一侧**，而是让你选。

**每个引用都能回溯。** 文本分块时记录了它来自哪篇论文的哪一页哪一节，
所以回答里的引用可以点击跳回 PDF 对应位置。

**AI 产出默认是草稿。** 自动标签进「建议队列」等人确认，深度阅读产出的笔记
标记为 draft。误合并两篇论文或让幻觉污染笔记库，代价远高于多点几下。

**零外部服务。** 没有 Redis、没有 Celery、没有独立向量库。SQLite 一个文件
装下元数据、全文索引（FTS5）和向量索引（sqlite-vec）。后台任务用数据库当队列。

---

## 模型配置

支持两类服务商：**Anthropic Messages API**（官方或兼容端点）与 **OpenAI 兼容接口**。
在「设置 → 模型」里配置，或者复用 Claude Code 已有的配置：

```bash
pipenv run flask --app wsgi kb llm import-claude-config --dry-run   # 先预览
pipenv run flask --app wsgi kb llm import-claude-config             # 再导入
pipenv run flask --app wsgi kb llm test                             # 验证连通
pipenv run flask --app wsgi kb llm capabilities --probe             # 实测能力
```

导入会读取 `~/.claude.json` 里 `env` 段的 `ANTHROPIC_*` 字段（含 API Key）。

### 兼容端点会「接受」它不支持的参数

这是使用第三方兼容端点时最容易踩的坑：**参数不报错，不等于生效**。

实测 DeepSeek 的 Anthropic 兼容端点：

| 能力 | 表现 |
|---|---|
| 基本对话 / system / 流式 | 正常 |
| 工具调用（含严格 schema） | 正常 |
| 图片输入 | 正常 |
| 提示缓存 | 正常（**自动生效**，第二次起命中） |
| PDF 文档块 | 接受，但**不解析**，模型答「无法确定」 |
| `output_config.format` 结构化输出 | 接受，但**输出被整个吞掉**，只剩思考 |

所以本系统不靠「没报错」判断能力，而是实测：`kb llm capabilities --probe`
会真的发请求验证，结果存进设置供后续复用。业务层据此选择路径——
不支持原生结构化输出就走工具调用，不支持 PDF 就本地解析后注入文本。

另外两个实测得到的约束：

* **思考与正文共用输出 token 额度。** 推理型模型回答一个是非题也会消耗两千多
  字符的思考；额度给小了会导致正文一个字都出不来。默认 16000。
* **不能用 `content[0].text` 取回复。** 这类端点默认返回 thinking 块，
  必须先按类型筛选——本系统的 Provider 层已经统一处理。

---

## 环境说明（实测结论）

以下几点是实际部署时验证过的，能省掉一些排查时间：

| 事项 | 结论 |
|---|---|
| 中文检索 | 用 SQLite 内置的 FTS5 `trigram` 分词器，**不需要 jieba**。3 字以上的中文查询走索引；1–2 字的短词（「模型」「点积」这类很常见）自动降级为 `LIKE` 扫描——实测 2 万分块下约 2.6 ms，可接受 |
| WAL | 在 9p 挂载（WSL 的 `/mnt/*`、`D:\`）上**实测可用**。默认仍把数据库放在 Linux 原生盘，因为 9p 的 IO 明显更慢 |
| 依赖安装 | 全部依赖都有预编译 wheel，**不需要编译器**。这在没有 gcc 的环境里是硬约束 |
| 启动自检 | 每次启动检查日志模式、FTS5、trigram、向量扩展，结果显示在设置页与 `/api/v1/system/health`。任一项不可用时会明确告警并降级，而不是静默返回空结果 |

---

## 许可提示

PDF 解析默认使用 **PyMuPDF，其许可为 AGPL-3.0**。个人自用没有影响；
若要闭源分发本应用，需要替换为 MIT 许可的解析后端（代码里已预留切换点）。

其余依赖均为宽松许可。

---

## 目录结构

```
kb/
├── kb/
│   ├── config.py        启动级配置（三层：默认值 < config.toml < 环境变量）
│   ├── settings.py      运行时配置（存数据库，网页端可改，含密钥加密）
│   ├── sqlite.py        PRAGMA 调优、启动自检、向量扩展加载
│   ├── models/          数据模型
│   ├── services/        业务逻辑（网页端与接口共用）
│   ├── jobs/            数据库任务队列 + 进程内 worker
│   ├── api/             对外 REST 接口
│   └── web/             服务端渲染页面
├── docs/                设计与接口文档
└── scripts/             辅助脚本
```

数据目录（默认 `~/.local/share/kb/`）保存数据库、密钥、上传缓存与任务产物，
与代码完全分离。

---

## 常用命令

```bash
pipenv run flask --app wsgi kb selfcheck        # 环境与数据库自检
pipenv run flask --app wsgi kb info             # 配置与统计
pipenv run flask --app wsgi kb scan --wait      # 扫描论文目录
pipenv run flask --app wsgi kb index --wait     # 解析并建索引
pipenv run flask --app wsgi kb jobs             # 任务列表
pipenv run flask --app wsgi kb settings list    # 查看所有配置项
pipenv run flask --app wsgi kb backup ~/kb.bak  # 备份数据库
pipenv run flask --app wsgi kb routes           # 所有路由
```

生产环境用 gunicorn：

```bash
pipenv run gunicorn -w 1 --threads 8 -b 0.0.0.0:5000 wsgi:app
```

**注意 `-w 1`**：后台 worker 内嵌在 Web 进程里，多进程会各跑一份任务队列。
需要横向扩展时把 `KB_WORKER_EMBEDDED` 设为 `0`，另起 `flask kb worker` 专门跑任务。
