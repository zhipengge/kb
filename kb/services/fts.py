"""全文索引：两张 FTS5 表 + 同步触发器。

**为什么是两张表。** FTS5 的分词器是建表时定的，一张表只能用一个。
中文需要 trigram（免分词、支持子串命中），英文需要 unicode61（按词切分、
BM25 排序质量更好）。所以建两张，检索时都查、结果融合。

**为什么用 external content。** ``content='chunks'`` 让 FTS5 只存倒排索引、
不存原文，查询时回原表取文本。好处是同一份文本只存一次——
对几十兆的论文正文来说，这个差别很实在。代价是必须自己维护同步触发器，
删改行时要按 FTS5 的约定发 'delete' 指令（见下方触发器）。

**关于 rowid。** ``chunks`` 表的主键是 ULID 文本，但 SQLite 每张表都有
隐含的 rowid，FTS5 的外部内容模式就是基于它工作的。触发器里用
``new.rowid`` / ``old.rowid``，查询时用 ``fts.rowid = chunks.rowid`` 关联。
"""

from __future__ import annotations

import logging
import re

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..utils.sql import safe_identifier

log = logging.getLogger(__name__)

FTS_EN = "chunks_fts"
FTS_CJK = "chunks_fts_cjk"

# 建表语句。两张表结构相同，只是分词器不同。
_CREATE_TEMPLATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='{tokenizer}'
)
"""

# 触发器：chunks 表增删改时同步 FTS 索引。
#
# 注意 delete 的写法 —— 外部内容模式下不能直接 DELETE，
# 必须按 FTS5 约定的特殊行格式把「待删除的原文」喂给它，
# 否则索引里会留下永远不会被清理的幽灵词条。
_TRIGGERS = {
    "chunks_fts_ai": f"""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO {FTS_EN}(rowid, text) VALUES (new.rowid, new.text);
            INSERT INTO {FTS_CJK}(rowid, text) VALUES (new.rowid, new.text);
        END
    """,
    "chunks_fts_ad": f"""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO {FTS_EN}({FTS_EN}, rowid, text) VALUES ('delete', old.rowid, old.text);
            INSERT INTO {FTS_CJK}({FTS_CJK}, rowid, text) VALUES ('delete', old.rowid, old.text);
        END
    """,
    "chunks_fts_au": f"""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE OF text ON chunks BEGIN
            INSERT INTO {FTS_EN}({FTS_EN}, rowid, text) VALUES ('delete', old.rowid, old.text);
            INSERT INTO {FTS_EN}(rowid, text) VALUES (new.rowid, new.text);
            INSERT INTO {FTS_CJK}({FTS_CJK}, rowid, text) VALUES ('delete', old.rowid, old.text);
            INSERT INTO {FTS_CJK}(rowid, text) VALUES (new.rowid, new.text);
        END
    """,
}


def ensure_fts(engine: Engine) -> dict:
    """创建 FTS 表与触发器。幂等，每次启动都调用。

    返回能力状态字典，供设置页展示。
    """
    state = {"enabled": False, "trigram": False, "tables": []}

    with engine.begin() as conn:
        # unicode61：英文按词切分
        try:
            conn.execute(text(_CREATE_TEMPLATE.format(name=FTS_EN, tokenizer="unicode61")))
            state["tables"].append(FTS_EN)
            state["enabled"] = True
        except Exception as exc:
            log.error("创建 FTS5 表 %s 失败：%s", FTS_EN, exc)

        # trigram：中文免分词子串检索
        try:
            conn.execute(text(_CREATE_TEMPLATE.format(name=FTS_CJK, tokenizer="trigram")))
            state["tables"].append(FTS_CJK)
            state["trigram"] = True
        except Exception as exc:
            log.warning(
                "创建 trigram 分词表失败：%s。中文检索将退化为 LIKE 扫描。", exc
            )

        for statement in _TRIGGERS.values():
            try:
                conn.execute(text(statement))
            except Exception as exc:
                log.error("创建 FTS 触发器失败：%s", exc)

    return state


def rebuild(engine: Engine) -> int:
    """重建全文索引。返回索引到的分块数。

    用 FTS5 的 'rebuild' 指令（比逐行插入快得多），它要求表以
    external content 模式创建——正是我们的情况。
    """
    with engine.begin() as conn:
        for table in (FTS_EN, FTS_CJK):
            try:
                conn.execute(text(f"INSERT INTO {table}({table}) VALUES('rebuild')"))
            except Exception as exc:
                log.error("重建 %s 失败：%s", table, exc)
        count = conn.execute(text("SELECT count(*) FROM chunks")).scalar() or 0
    log.info("全文索引重建完成，共 %d 个分块", count)
    return count


def optimize(engine: Engine) -> None:
    """合并 FTS 内部的段，减小体积、提升检索速度。

    大量写入之后调一次即可，日常检索不需要。
    """
    with engine.begin() as conn:
        for table in (FTS_EN, FTS_CJK):
            try:
                conn.execute(text(f"INSERT INTO {table}({table}) VALUES('optimize')"))
            except Exception as exc:
                log.warning("优化 %s 失败：%s", table, exc)


# --------------------------------------------------------------------------
# 查询构造
# --------------------------------------------------------------------------


def is_cjk(text_value: str) -> bool:
    return any("一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" for ch in text_value)


def min_trigram_len(text_value: str) -> int:
    """查询词里最长的连续 CJK 片段长度。

    trigram 分词器只对 **3 个字符以上**的片段建索引，所以「点积」这种
    两字词在 trigram 表里查不到任何东西——这不是 bug，是分词器的定义。
    调用方据此决定走索引还是走 LIKE 兜底。
    """
    longest = 0
    current = 0
    for ch in text_value:
        if "一" <= ch <= "鿿":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def extract_ascii_terms(query: str, *, min_length: int = 2) -> str:
    """从混排查询里挑出 ASCII 技术词。

    「变分自编码器的 ELBO 是怎么推导出来的」这种中文提问里，
    真正能匹配英文语料的是 ``ELBO`` 这个技术词，其余都是中文的表述架子。
    把它们挑出来单独检索，是中文提问命中英文论文最直接的办法。
    """
    terms = re.findall(rf"[A-Za-z][A-Za-z0-9._-]{{{min_length - 1},}}", query)
    # 去掉太常见的英文虚词，它们匹配面太宽
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "are", "was",
                 "how", "what", "why", "does", "can", "its", "it", "of", "in", "on"}
    kept = [t for t in terms if t.lower() not in stopwords]
    # 长词优先，通常也更有区分度
    kept.sort(key=len, reverse=True)
    return " ".join(kept[:6])


def build_match_query(query: str, operator: str = "AND") -> str | None:
    """把用户输入变成安全的 FTS5 MATCH 表达式。

    两条要求同时满足：

    **安全。** FTS5 的语法里 ``"``、``*``、``:``、``NEAR``、``NOT`` 都是操作符。
    原样透传用户输入，轻则语法错误（整个请求 500），重则被构造出代价极高的
    查询。这里把每个词单独加引号当字面量处理，引号本身翻倍转义，
    用户输入里不可能再出现操作符。

    **符合直觉。** 多词查询按 AND 组合而不是当成一个短语：搜
    「routing capacity」的人想要的是同时提到这两个词的内容，
    而不是要求它们必须相邻出现。整串加引号（短语匹配）会把绝大多数
    正常查询变成 0 结果——这是很容易踩进去的坑。

    CJK 片段整体作为一个字面量，交给 trigram 做子串匹配。
    """
    if not query or not query.strip():
        return None

    # 按空白与常见标点切词。不切 CJK——中文没有词间空格，
    # 切开反而破坏 trigram 的子串匹配
    parts = re.split(r"[\s,;、，。；：!?！？()（）\[\]{}<>\"'`/\\|]+", query.strip())

    terms: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 单个 ASCII 字母/数字没有检索价值，还会引入大量噪音
        if len(part) == 1 and part.isascii() and part.isalnum():
            continue
        terms.append(f'"{part.replace(chr(34), chr(34) * 2)}"')

    if not terms:
        return None
    return f" {operator} ".join(terms)


# 兼容旧名字
escape_match_query = build_match_query


def cjk_ngrams(text_value: str, *, n: int = 3, limit: int = 40) -> list[str]:
    """把中文长句切成重叠的 n 元组，供整句匹配不到时放宽使用。

    **为什么需要这一步。** trigram 分词器把中文按 3 字符建索引，而
    ``build_match_query`` 有意不切 CJK——整句当一个字面量交给子串匹配。
    对「扩散模型」这种短词没问题，但「扩散模型的训练目标是什么」这 12 个字
    在语料里当然不会原样出现，于是：

      * AND 匹配：0 条；
      * OR 匹配：整句仍是一个字面量，还是 0 条；
      * 只查 ASCII 技术词：这句话里一个 ASCII 词都没有，还是 0 条。

    三步走完返回空。**在查询扩展（LLM）不可用时，这类问句一条都搜不到**——
    界面上显示「知识库里没有找到相关内容」，而库里明明有扩散模型的论文。
    实测概念类问句 5 条全部零结果，直到补上这一步。

    切出来的是 n 元组而不是词，因为中文分词需要一个词典，而这里没有。
    噪声（「的训练」这类）由 bm25 的 IDF 权重自然压低：越常见的组合
    贡献越小。这个函数只在**其它通道全部落空**时才被调用，此时
    「有一点召回」总好过「零结果」。

    n 取 3 是为了对齐 trigram 分词器的索引单位——查一个 3 字片段
    正好命中一个倒排项。短于 3 字的串原样返回（退化成整串匹配，
    由调用方的 LIKE 兜底）。
    """
    if not text_value:
        return []

    grams: list[str] = []
    seen: set[str] = set()
    for run in re.findall(r"[㐀-䶿一-鿿]+", text_value):
        if len(run) < n:
            # 太短，整串当一个词——trigram 表里查不到，但 LIKE 兜底用得上
            if run not in seen:
                seen.add(run)
                grams.append(run)
            continue
        for start in range(len(run) - n + 1):
            gram = run[start : start + n]
            if gram in seen:
                continue
            seen.add(gram)
            grams.append(gram)
            if len(grams) >= limit:
                return grams
    return grams


def build_search_sql(table: str, *, limit: int, extra_where: str = "") -> str:
    """构造对某张 FTS 表的检索语句。

    ``bm25()`` 返回的是**负**数（越小越相关），排序时用 ASC；
    这里换成正值并用 DESC，避免调用方每次都要想一下方向。

    表名无法参数化，只能拼进语句，所以先过一遍标识符校验（见 utils/sql.py）。
    """
    safe_identifier(table)
    return f"""
        SELECT chunks.id            AS chunk_id,
               chunks.paper_id      AS paper_id,
               chunks.note_id       AS note_id,
               chunks.section_path  AS section_path,
               chunks.page_from     AS page_from,
               chunks.page_to       AS page_to,
               chunks.kind          AS kind,
               chunks.text          AS text,
               -bm25({table})       AS score
          FROM {table}
          JOIN chunks ON chunks.rowid = {table}.rowid
         WHERE {table} MATCH :query
           {extra_where}
         ORDER BY score DESC
         LIMIT :limit
    """


__all__ = [
    "FTS_CJK",
    "FTS_EN",
    "build_match_query",
    "build_search_sql",
    "cjk_ngrams",
    "ensure_fts",
    "escape_match_query",
    "is_cjk",
    "min_trigram_len",
    "optimize",
    "rebuild",
]
