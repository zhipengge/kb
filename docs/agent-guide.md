# 论文知识库 · 外部 agent 使用文档

这份文档写给**要调用这个知识库的外部 agent**。读完即可上手，不需要读源码。

知识库里是一个人的私人论文库（论文原文 + AI 精读笔记 + 关联代码仓库），
支持混合检索、带出处的问答、以及写笔记/标签。

---

## 0. 先选接入方式

**如果你支持 MCP，用 MCP。** 工具清单与语义都是现成的，不必自己拼 HTTP。

| | MCP | REST |
|---|---|---|
| 适合 | 所有读取场景 | 不支持 MCP，或要写数据 |
| 工具/端点 | 10 个工具，**全部只读** | 全部端点 |
| 需要服务先跑起来吗 | **stdio 模式不需要**（客户端自己拉起子进程，直接读同一个数据库） | 需要 |

MCP 工具（全部只读）：
`search` `read_paper` `ask` `list_papers` `get_paper` `get_note`
`list_notes` `list_tags` `get_code` `stats`

```bash
# stdio：客户端自己拉起，不需要 Web 服务在跑
claude mcp add kb -- "$(pipenv --venv)/bin/python" "$PWD/scripts/mcp_server.py"
```

HTTP 模式则先启动服务，再 `POST <base>/api/v1/mcp`（JSON-RPC 2.0，需 API Key）。

下面讲 REST。**MCP 工具的语义与之一一对应**，读完同样适用于 MCP 调用。

---

## 1. 鉴权

```bash
curl -H "Authorization: Bearer kb_xxxxxxxx_..." <base>/api/v1/system/ping
```

- Key 由知识库主人在网页设置页或 `flask kb key create --scope read` 签发，**只显示一次**。
- 作用域：`read` / `write` / `ingest` / `admin`。只读取材料用 `read` 就够。
- **免鉴权的只有三个**：`/api/v1/openapi.json`、`/api/v1/agent-guide`、
  `/api/v1/system/ping`。其余一律 401。

> 注意：`/api/v1/system/health` **需要鉴权**，尽管它看起来像健康检查。

### 响应信封

```jsonc
// 成功
{"ok": true, "data": ..., "meta": {"request_id": "...", ...}}
// 失败
{"ok": false, "error": {"code": "...", "message": "...", "details": {}},
 "meta": {"request_id": "..."}}
```

出错时把 `meta.request_id` 带上再反馈，服务端能直接定位到那一条日志。

错误码：`unauthorized` `forbidden` `not_found` `invalid_argument`
`rate_limited` `llm_error` `conflict`

---

## 2. 核心范式：search → read → cite

这三步是整个接口的设计中心，按它走就不会错。

### 第一步：检索，拿到片段和精确定位

```bash
POST <base>/api/v1/search
{"q": "world model autonomous driving", "limit": 3}
```

每条结果：

```jsonc
{
  "chunk_id": "01M48W0P9K1MJP2R2SSMDDP2HY",
  "paper_id": "01M48369K3PTKH67QEY8TAN4DW",
  "paper_title": "HyWorldVLA: ...",
  "note_id": null,                 // 非 null 表示这块来自「笔记」而不是论文原文
  "kind": "text",                  // text | figure | table | formula | code | note
  "locator": "§Experiments > Ablation Analysis p.6",
  "section_path": "Experiments > Ablation Analysis",
  "page_from": 6, "page_to": 6,
  "score": 0.041,
  "snippet": "…带省略号的摘要…",
  "text": "该分块的完整正文……"      // 注意：是全文，见 §4
}
```

`locator` 已经拼好，**直接拿来当出处写进你的回答**，不要自己拼页码。

### 第二步：要更多上下文时按定位取

```bash
GET <base>/api/v1/papers/<paper_id>/text                  # 不传参 → 全篇目录 + 开头几块
GET <base>/api/v1/papers/<paper_id>/text?section=Ablation # 按小节名子串匹配
GET <base>/api/v1/papers/<paper_id>/text?page=6           # 按页
GET <base>/api/v1/papers/<paper_id>                       # 元数据
GET <base>/api/v1/notes?paper_id=<paper_id>               # 该论文的精读笔记
```

`data` 形如 `{paper_id, title, outline, chunks[], total_chunks, offset, returned, next_offset}`。
用 `offset` / `next_offset` 翻页，不要一次拉全篇。

**笔记值得优先看。** 它是已经消化过的中文内容（方法梳理、局限、复现要点），
比论文原文的信息密度高得多；`note_id` 非空的检索结果就是它。

### 第三步：作答并标注出处

```bash
POST <base>/api/v1/ask
{"question": "NAVSIM 的 PDMS 指标怎么算？", "limit": 3}
```

它内部完成检索 + 生成 + **引文核对**：

```jsonc
{
  "answer": "……[1][3]……",
  "citations": [
    {"marker": 1, "check": "verified",
     "quote": "we use the PDM Score (PDMS), which comprises ...",
     "locator": "§Experiments > Datasets and metrics p.6",
     "paper_id": "...", "chunk_id": "...", "page_from": 6, "page_to": 6}
  ],
  "grounded": true,
  "model": "...", "tokens_used": 2949, "elapsed_ms": 12300,
  "web_citations": []
}
```

---

## 3. 引用可信度：**必须看 `check` 字段**

这是本接口最重要的一个语义。答案里的 `[1]` `[2]` 与 `citations[].marker` 一一对应，
每条带 `check` 三态：

| `check` | 含义 | 你该怎么做 |
|---|---|---|
| `verified` | 引文能在被引分块里**逐字找到** | 可以放心引用 |
| `unverified` | 模型**没给**可核对的引文（常见于笔记类引用） | 引用前自己看一眼原文 |
| `mismatched` | 给了引文但**对不上** | **不要引用**，多半是模型编的 |

`grounded: false` 表示至少有一条 `mismatched`。

**不要把三态简化成「可信 / 不可信」。** `unverified` 只是没用上校验能力，
`mismatched` 才意味着内容可能是编的——两者混为一谈会让你对校验结果失去分辨力。

### 联网来源是**另一回事**

知识库里没有答案时，服务端可能自己联网检索。结果放在**单独的 `web_citations`**：

| | 来源 | 校验 | marker |
|---|---|---|---|
| `citations` | 知识库 | 有三态 `check` | `1` `2` |
| `web_citations` | 互联网 | **没有校验** | `W1` `W2` |

**两者可信度不是一个量级。** 不要混用，也不要给 `web_citations` 加上
「已验证」之类的说法。每条带 `url` / `kind`(paper\|code\|web) / `source` / `published`，
可以自己点开核对。

---

## 4. 成本与上下文：`search` **不是免费的**

这一条最容易被低估。

- **`search` 会触发一次查询扩展的大模型调用**（把中文提问翻成英文术语），
  实测约 1.6 秒、几百 token。**同一个查询词有缓存**，重复查不再花钱。
  想省钱就少发几次、每次问得具体些——**缩小 `limit` 并不能省掉这一步**。
- **`ask` 比 `search` 贵一到两个数量级**，也慢得多（含思考通常十几秒）。
  一次 `ask` 内部也会做一次检索，所以它包含 `search` 的那笔开销。
- 其余工具/端点不调用模型：`list_papers` `get_paper` `get_note` `read_paper`
  `stats` `list_tags` 等。

**`search` 返回的 `text` 是分块的完整正文，不是摘要。** 一个块几百到上千 token
很正常，`limit: 50` 会直接把你的上下文撑爆。默认 8 条已经不少，
**建议从 `limit: 3~5` 起步**，不够再加。`limit` 上限 100。

---

## 5. 常见坑

**不要在检索词里用搜索引擎语法。** `search` 接的是结构化 API
（OpenAlex / Crossref / arXiv / GitHub / HackerNews），不是搜索引擎。
`site:`、引号、`OR`、`-` 这类操作符它们不认，**会把结果清空**。
实测同一个问题：平实关键词能出 4 类来源，加上 `site:news.ycombinator.com`
就只剩 1 类。

**用英文检索词效果最好。** 学术库和 GitHub 都以英文为主。中文提问也能命中
（服务端会做跨语言扩展），但英文术语更准。

**别问需要推理的问题给 `search`。** 检索是字面/语义匹配，不做推理。
「这两篇方法有什么不同」这类先 `search` 拿材料再自己判断，或者直接用 `ask`。

**`limit` 与 `mode` 在查询串和 POST 请求体里都认**，两种写法等价：

```bash
GET  <base>/api/v1/search?q=world+model&limit=3
POST <base>/api/v1/search   {"q": "world model", "limit": 3}   # 两者等价
```

**分页**：列表端点用 `limit` + 不透明游标，下一页游标在 `meta.next_cursor`，
原样回传即可，不要自己解析。

---

## 6. 常用端点速查

| 端点 | 用途 |
|---|---|
| `POST /api/v1/search` | 检索分块（**入口**，返回 `locator`） |
| `GET /api/v1/search/papers` | 按论文聚合的检索（「找几篇相关的看看」） |
| `POST /api/v1/ask` | 带引用校验的问答（**贵**） |
| `GET /api/v1/papers` | 论文列表。过滤：`q` `year` `venue` `reading_status` `ingest_status` `has_code` `tag`（可重复，AND） |
| `GET /api/v1/papers/<id>` | 论文元数据 |
| `GET /api/v1/papers/<id>/text` | 正文分块（按页/按小节） |
| `GET /api/v1/papers/<id>/file` | PDF 原文，支持 Range |
| `GET /api/v1/notes?paper_id=<id>` | 笔记 |
| `GET /api/v1/tags` | 标签词表 |
| `GET /api/v1/search/stats` | 检索配置与统计 |

**写操作**（需 `write` 及以上作用域的 Key）：

| 操作 | 端点 |
|---|---|
| 新建笔记 | `POST /api/v1/notes` |
| 改 / 删笔记 | `PATCH`\|`PUT`\|`DELETE /api/v1/notes/<id>` |
| 改 / 删论文 | `PATCH`\|`DELETE /api/v1/papers/<id>`（**没有 PUT**） |
| 给论文加标签 | `POST /api/v1/papers/<id>/tags` |
| 回滚笔记到历史版本 | `POST /api/v1/notes/<id>/restore/<version>` |

注意集合端点（`/notes`、`/papers`）**只支持 GET 和 POST**，
删改一律要带 ID 的具体资源端点。

改笔记走**乐观锁**：请求里带上你读到的 `version`，不匹配会返回 `conflict`，
此时重新读取再改，不要盲目重试。

---

## 7. 想读更细的：让服务自己说

```bash
GET <base>/api/v1/agent-guide     # 纯文本自述，可直接塞进上下文，不需要鉴权
GET <base>/api/v1/openapi.json    # OpenAPI 3.1，路径从真实路由自动生成，不需要鉴权
```

`openapi.json` 里的路径是**从 Flask 路由表自动生成**的，所以「文档里有、实际 404」
不会发生；反过来，没人工补说明的端点会带 `x-undocumented: true`——
看到这个标记就说明该端点确实存在、只是没写用途。

---

## 8. 出错时的判断顺序

1. **401** → Key 没带、写错、或已被吊销。
2. **403** → Key 的 scope 不够（比如用 `read` Key 去写笔记）。
3. **400 `invalid_argument`** → 缺 `q`/`question` 这类必填字段，看 `error.message`。
4. **`search` 返回空** → 先确认检索词里没有搜索引擎操作符（§5），再试英文术语。
   **注意区分「知识库里没有」和「检索坏了」**：后者会在 `meta` 或错误里体现，
   不要一律当成「没有」。
5. **`llm_error`** → 模型侧问题（额度、限流、网关）。`search` 的部分结果可能仍然可用。
6. **`conflict`**（写操作）→ 乐观锁版本不匹配，重新读取再改。
