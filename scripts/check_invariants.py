#!/usr/bin/env python
"""检查那些「不报错但会让人用错」的不变量。

为什么需要这个脚本，而不是单元测试：这里查的都是**跨查询的一致性**——
两个查询各自都正确，但口径不一致时就出问题。典型例子：

    侧栏徽章显示「自动驾驶 59 篇」，点进去却是 0 篇。

这不会抛异常、不会写日志、接口返回 200，只是默默地把人引向错误的结论。
单元测试很难覆盖（要为每个查询各写一份期望值），而这类一致性在真实数据上
一验就露馅。所以做成可以随时跑的脚本：

    pipenv run python scripts/check_invariants.py

退出码非 0 表示有不变量被破坏。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb import create_app
from kb.extensions import db
from kb.models import Note, NoteRevision, NoteTag, Paper, PaperTag, Tag
from kb.services import papers as papers_service

problems: list[str] = []


def _check_facets_match_filters() -> int:
    """分面计数必须等于按该标签筛选出来的篇数。

    这是最容易被破坏的一条：计数和筛选是两套 SQL，一旦口径分叉
    （比如一边数关联表行数、一边按论文去重），数字就对不上。
    """
    checked = 0
    for tags in papers_service.tag_facets().values():
        for tag in tags:
            if tag["count"] == 0:
                continue
            _, _, total = papers_service.list_papers(tag_ids=[tag["id"]])
            checked += 1
            if total != tag["count"]:
                problems.append(
                    f"标签分面计数与筛选结果不一致：{tag['name']!r} "
                    f"分面显示 {tag['count']}，实际筛出 {total}"
                )
    return checked


def _check_tag_attachment() -> int:
    """至少要有一条标签关联存在。

    这里查的是「哪种挂载方式真的在用」：历史上 paper_tags 一行都没有，
    所有标签都挂在笔记上，而筛选只查 paper_tags —— 功能整体失效但不报错，
    界面上还显示着漂亮的使用计数。把它查出来，是为了让这种静默失效
    在下一次发生时立刻暴露。
    """
    paper_links = db.session.query(PaperTag).count()
    note_links = db.session.query(NoteTag).count()
    if paper_links == 0 and note_links == 0:
        problems.append("标签一条都没挂上：paper_tags 与 note_tags 都是空的")
    return paper_links + note_links


def _check_note_version_monotonic() -> None:
    """笔记当前版本必须大于它所有历史快照的版本号。

    快照在改动前拍下、版本号随后自增。若某次改动漏了自增，
    回滚会拿到「和当前一模一样」的内容——版本历史静默失效。
    """
    rows = (
        db.session.query(Note.id, Note.version, db.func.max(NoteRevision.version))
        .join(NoteRevision, NoteRevision.note_id == Note.id)
        .group_by(Note.id)
        .all()
    )
    for note_id, current, highest in rows:
        if current is not None and highest is not None and current <= highest:
            problems.append(
                f"笔记 {note_id} 当前版本 {current} 不大于最新快照 {highest}，"
                "回滚会拿到与当前相同的内容"
            )


def _check_orphan_notes() -> int:
    """笔记指向的论文必须存在（外键被绕过时会出现悬空引用）。"""
    orphans = (
        db.session.query(Note)
        .filter(Note.paper_id.isnot(None))
        .outerjoin(Paper, Paper.id == Note.paper_id)
        .filter(Paper.id.is_(None))
        .count()
    )
    if orphans:
        problems.append(f"有 {orphans} 篇笔记指向了不存在的论文")
    return orphans


def _check_agent_docs_match_routes() -> int:
    """agent 指南里提到的每个路径都必须真实存在。

    这条检查是**为了一类已经发生过的缺陷**：guide 里写着
    ``GET /api/v1/papers/<id>/text``，而那个端点压根没实现——机器按指南调
    只会拿到 404，然后合理地判断「这个知识库没有这个能力」。

    对外文档和路由漂移是必然会发生的：改路由的人不会想起还有一份纯文本说明
    散在别处。与其指望人记得，不如让它在自检里直接失败。
    """
    import re

    from flask import current_app

    from kb.api.docs import agent_guide

    routes = {str(rule.rule) for rule in current_app.url_map.iter_rules()}
    with current_app.test_request_context("/api/v1/agent-guide"):
        text = agent_guide().get_data(as_text=True)

    mentioned = {m.rstrip(".") for m in re.findall(r"/api/v1/[A-Za-z0-9_/<>.\-]+", text)}
    missing = []
    for path in sorted(mentioned):
        if path in routes:
            continue
        # 参数名可能不同（<paper_id> vs <id>），按去掉占位符后的形状比对
        shape = re.sub(r"<[^>]+>", "{}", path)
        if any(re.sub(r"<[^>]+>", "{}", r) == shape for r in routes):
            continue
        missing.append(path)

    if missing:
        problems.append(
            f"agent 指南提到了不存在的路径：{', '.join(missing)}"
        )
    return len(mentioned)


def _check_static_agent_doc() -> int:
    """``docs/agent-guide.md`` 里提到的端点都必须真实存在，**且方法对得上**。

    上面那条检查只覆盖**动态**的 agent-guide 端点（服务端渲染的纯文本）。
    而仓库里还有一份静态文档——它是给「服务还没跑起来」或「要直接塞进模型
    上下文」的场景用的，同样会漂移，而且更容易：改路由的人根本不知道
    md 文件里还抄了一份。

    比动态那份多查一件事：**HTTP 方法**。写文档时实测踩到过——
    路径全对，但把 ``PATCH /papers/<id>`` 写成了 ``PUT``，把集合端点
    ``DELETE /notes`` 当成了存在（DELETE 只在 ``/notes/<id>`` 上）。
    路径检查抓不到这两种错，而照着调的调用方会拿到 405 或 404。
    """
    import re
    from pathlib import Path

    from flask import current_app

    doc = Path(current_app.root_path).parent / "docs" / "agent-guide.md"
    if not doc.exists():
        problems.append("docs/agent-guide.md 不存在（外部 agent 的使用文档）")
        return 0

    text = doc.read_text(encoding="utf-8")
    by_shape: dict[str, set[str]] = {}
    for rule in current_app.url_map.iter_rules():
        shape = re.sub(r"<[^>]+>", "{}", str(rule.rule))
        by_shape.setdefault(shape, set()).update(rule.methods - {"HEAD", "OPTIONS"})

    # 先把 Markdown 的包裹符号去掉，否则方法名和路径会被它们隔开。
    #
    # **这一步不能省。** 表格里的写法是 `` `PUT`\|`DELETE /api/v1/...` ``，
    # 反引号和 `\|` 夹在 `PUT` 与路径之间；正则要求方法与路径相邻，
    # 于是 `PUT` 会被静默丢掉，只校验到 `DELETE`——而 DELETE 恰好是支持的，
    # 检查就「通过」了。第一版就是这么写的，注入 `PUT /papers/<id>` 这种
    # 真实错误时它毫无反应。**一个在常见写法下静默失效的守卫比没有更糟**：
    # 它会让人以为这块被盯住了。
    normalized = text.replace("`", "").replace("\\|", "|")

    checked = 0
    pattern = re.compile(
        r"((?:GET|POST|PUT|PATCH|DELETE)(?:\s*[|/]\s*(?:GET|POST|PUT|PATCH|DELETE))*)?"
        r"[\s|/]*?(/api/v1/[A-Za-z0-9_/<>.\-]+)",
    )
    for methods_str, path in pattern.findall(normalized):
        path = path.rstrip(".,)")
        shape = re.sub(r"<[^>]+>", "{}", path)
        have = by_shape.get(shape)
        if have is None:
            problems.append(f"docs/agent-guide.md 提到了不存在的路径：{path}")
            continue
        checked += 1
        if not methods_str:
            continue
        for method in re.findall(r"GET|POST|PUT|PATCH|DELETE", methods_str):
            if method not in have:
                problems.append(
                    f"docs/agent-guide.md 写的是 {method} {path}，"
                    f"但该端点实际只支持 {', '.join(sorted(have))}"
                )
    return checked


def _check_mcp_tool_schemas() -> int:
    """MCP 工具表必须自洽。

    工具是模型唯一的说明书，schema 写错了它只能瞎试。这里只钉住几条
    结构性的硬要求——required 里的字段必须在 properties 里有定义，
    JSON Schema 里这是硬错误，客户端可能直接拒收整个 tools/list。
    """
    from kb.services.agent_tools import TOOLS

    problems_before = len(problems)
    for tool in TOOLS:
        schema = tool.input_schema
        for name in schema.get("required", []):
            if name not in schema["properties"]:
                problems.append(f"工具 {tool.name} 的 required 字段 {name} 没有定义")
        if not tool.description.strip():
            problems.append(f"工具 {tool.name} 没有描述——模型只能靠它判断何时该用")
    if len(problems) != problems_before:
        return len(TOOLS)
    return len(TOOLS)


def main() -> None:
    app = create_app()
    with app.app_context():
        papers = db.session.query(Paper).filter(Paper.deleted_at.is_(None)).count()
        notes = db.session.query(Note).count()
        tags = db.session.query(Tag).count()
        print(f"数据：{papers} 篇论文 · {notes} 篇笔记 · {tags} 个标签\n")

        checked = _check_facets_match_filters()
        print(f"  ✓ 标签分面一致性（核对了 {checked} 个有论文的标签）"
              if checked else "  – 没有带论文的标签，跳过分面一致性检查")

        links = _check_tag_attachment()
        print(f"  ✓ 标签挂载（论文级 + 笔记级共 {links} 条关联）")

        _check_note_version_monotonic()
        print("  ✓ 笔记版本号单调递增")

        _check_orphan_notes()
        print("  ✓ 笔记没有悬空引用")

        mentioned = _check_agent_docs_match_routes()
        print(f"  ✓ agent 指南里的 {mentioned} 个路径都真实存在")

        static_checked = _check_static_agent_doc()
        print(f"  ✓ 静态使用文档里的 {static_checked} 个端点与方法都真实存在")

        tools = _check_mcp_tool_schemas()
        print(f"  ✓ MCP 工具表自洽（{tools} 个工具）")

        print()
        if problems:
            print(f"发现 {len(problems)} 个问题：")
            for item in problems:
                print(f"  ✗ {item}")
            sys.exit(1)
        print("全部通过。")


if __name__ == "__main__":
    main()
