"""网页端路由。

视图函数只做三件事：取数据、渲染模板、把表单提交交给服务层。
业务逻辑一律不写在这里——同一个操作在接口里也要能用，两边共用 services。
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import (
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from ..extensions import db
from ..models import Chunk, Job, Note, Paper, Tag
from ..settings import BY_KEY, GROUPS, SettingsError, grouped_definitions
from . import web_bp

log = logging.getLogger(__name__)


@web_bp.get("/")
def dashboard():
    """概览：分类汇总 + 待办 + 最近活动。"""
    settings = current_app.extensions["kb_settings"]
    from ..services import stats as stats_service

    # 统计口径全部收拢到 services/stats.py。以前是四个 count() 内联在视图里，
    # 别处要用就又写一遍，两边的口径迟早会分叉（比如漏掉 reviewed 状态）。
    summary = stats_service.collect()
    stats = summary["overview"]

    recent_papers = (
        db.session.query(Paper)
        .filter(Paper.deleted_at.is_(None))
        .order_by(Paper.created_at.desc())
        .limit(8)
        .all()
    )

    active_jobs = (
        db.session.query(Job)
        .filter(Job.status.in_(("queued", "running")))
        .order_by(Job.created_at.desc())
        .limit(5)
        .all()
    )
    recent_jobs = (
        db.session.query(Job).order_by(Job.created_at.desc()).limit(5).all()
    )

    # 待办：让用户一眼看到「有什么在等我处理」
    from ..models import PaperDuplicate, TagSuggestion

    todos = {
        "suggestions": db.session.query(TagSuggestion)
        .filter(TagSuggestion.status == "pending")
        .count(),
        "duplicates": db.session.query(PaperDuplicate)
        .filter(PaperDuplicate.status == "pending")
        .count(),
        "failed_jobs": db.session.query(Job).filter(Job.status == "failed").count(),
    }

    # 路径可用性提示：根目录不存在时用户需要立刻知道，
    # 而不是等扫描任务跑完发现一篇都没扫到
    from ..services.paths import check_roots

    root_status = check_roots(settings.papers_roots)
    missing_roots = [r for r in root_status if not r["exists"]]

    return render_template(
        "dashboard.html",
        stats=stats,
        summary=summary,
        recent_papers=recent_papers,
        active_jobs=active_jobs,
        recent_jobs=recent_jobs,
        todos=todos,
        missing_roots=missing_roots,
        preflight=current_app.extensions.get("kb_preflight"),
    )


@web_bp.get("/settings")
def settings_page():
    """设置页。表单由 settings 的 schema 渲染，加配置项不需要改模板。"""
    settings = current_app.extensions["kb_settings"]
    cfg = current_app.extensions["kb_boot_config"]
    report = current_app.extensions.get("kb_preflight")
    vec_state = current_app.extensions.get("kb_vector_state", {})

    from ..services.llm import configuration_status
    from ..services.paths import check_roots

    values = settings.all_with_source()
    root_status = {
        "papers": check_roots(settings.papers_roots),
        "notes": check_roots([settings.notes_root] if settings.notes_root else []),
        "codes": check_roots([settings.codes_root] if settings.codes_root else []),
    }

    return render_template(
        "settings.html",
        values=values,
        groups=GROUPS,
        grouped=grouped_definitions(),
        preflight=report,
        vector_state=vec_state,
        boot_config=cfg,
        root_status=root_status,
        llm_status=configuration_status(),
        claude_config_path=str(Path.home() / ".claude.json"),
        claude_config_exists=(Path.home() / ".claude.json").is_file(),
        active_tab=request.args.get("tab", "paths"),
    )


@web_bp.post("/settings")
def settings_save():
    """保存设置。只保存当前 tab 提交上来的字段。"""
    settings = current_app.extensions["kb_settings"]
    tab = request.form.get("_tab", "paths")

    submitted = {}
    for key, value in request.form.items():
        if key.startswith("_") or key not in BY_KEY:
            continue
        submitted[key] = value

    # 复选框未勾选时浏览器不会提交该字段，需要把本 tab 内的布尔项补成 False
    for definition in grouped_definitions().get(tab, []):
        if definition.type == "bool" and definition.key not in submitted:
            submitted[definition.key] = False

    try:
        settings.update_many(submitted, updated_by="web")
    except SettingsError as exc:
        flash(str(exc), "error")
        return redirect(url_for("web.settings_page", tab=tab))

    flash("设置已保存", "ok")
    return redirect(url_for("web.settings_page", tab=tab))


@web_bp.post("/settings/create-roots")
def create_roots():
    """一键创建缺失的默认目录。"""
    from ..services.paths import ensure_default_dirs

    created, errors = ensure_default_dirs(current_app.extensions["kb_settings"])
    if errors:
        for message in errors:
            flash(message, "error")
    if created:
        flash(f"已创建 {len(created)} 个目录：" + "、".join(created), "ok")
    elif not errors:
        flash("所有目录都已存在", "ok")
    return redirect(url_for("web.settings_page", tab="paths"))


@web_bp.post("/settings/llm/import-claude-config")
def settings_import_claude_config():
    """从 Claude Code 的配置文件导入模型设置。

    这是显式操作：它会读取用户主目录下的密钥并写进知识库，
    不能在启动时悄悄做掉。
    """
    from ..services.llm import bootstrap

    settings = current_app.extensions["kb_settings"]
    try:
        result = bootstrap.import_from_claude_config(settings, updated_by="web")
        flash(
            f"已导入：{result['base_url']} · {result['deep_model']}"
            f"（认证方式 {result['auth_mode']}）",
            "ok",
        )
    except bootstrap.ClaudeConfigError as exc:
        flash(str(exc), "error")
    except Exception as exc:
        log.exception("导入 Claude Code 配置失败")
        flash(f"导入失败：{exc}", "error")
    return redirect(url_for("web.settings_page", tab="llm"))


@web_bp.post("/settings/llm/test")
def settings_test_llm():
    """发一条最小请求验证模型配置。"""
    from ..services.llm import get_provider

    try:
        provider = get_provider()
        response = provider.complete(
            [{"role": "user", "content": "只回复两个字：收到"}], max_tokens=2000
        )
        detail = f"模型 {response.model}"
        if response.thinking:
            detail += f" · 思考 {len(response.thinking)} 字符"
        detail += (
            f" · 用量 {response.usage.input_tokens}/{response.usage.output_tokens}"
        )
        flash(f"连接正常：{detail}", "ok")
    except Exception as exc:
        log.exception("模型连通性测试失败")
        flash(f"连接失败：{exc}", "error")
    return redirect(url_for("web.settings_page", tab="llm"))


@web_bp.post("/settings/llm/probe")
def settings_probe_llm():
    """实测服务商能力并保存结果。

    会发多次真实请求、消耗额度，所以是显式按钮而不是自动执行。
    """
    from ..services.llm import get_provider, save_capabilities
    from ..services.llm.probe import probe_capabilities

    settings = current_app.extensions["kb_settings"]
    try:
        provider = get_provider()
        report = probe_capabilities(provider)
        if report.get("fatal"):
            flash(f"探测失败：{'；'.join(report.get('errors') or ['未知错误'])}", "error")
        else:
            caps = report["capabilities"]
            save_capabilities(settings, provider, caps)
            supported = [
                label
                for key, label in (
                    ("pdf_native", "PDF 解析"), ("vision", "图片"), ("prompt_cache", "提示缓存"),
                    ("structured_output", "结构化输出"), ("tool_use", "工具调用"),
                )
                if caps.get(key)
            ]
            flash(f"探测完成。支持：{'、'.join(supported) or '（仅基本对话）'}", "ok")
    except Exception as exc:
        log.exception("能力探测失败")
        flash(f"探测失败：{exc}", "error")
    return redirect(url_for("web.settings_page", tab="llm"))


@web_bp.get("/jobs")
def jobs_page():
    """任务列表页。"""
    status = request.args.get("status")
    query = db.session.query(Job)
    if status:
        query = query.filter(Job.status == status)
    jobs = query.order_by(Job.created_at.desc()).limit(100).all()

    counts = {
        s: db.session.query(Job).filter(Job.status == s).count()
        for s in ("queued", "running", "succeeded", "failed", "cancelled")
    }
    return render_template("jobs.html", jobs=jobs, counts=counts, status=status)


@web_bp.post("/jobs/<job_id>/cancel")
def cancel_job_view(job_id: str):
    job = db.session.get(Job, job_id)
    if job is None:
        flash("任务不存在", "error")
    elif job.is_terminal:
        flash(f"任务已经结束（{job.status}）", "error")
    else:
        job.cancel_requested = True
        db.session.commit()
        flash("已请求取消", "ok")
    return redirect(url_for("web.jobs_page"))


@web_bp.post("/jobs/<job_id>/retry")
def retry_job_view(job_id: str):
    """重跑一个任务：复制参数入队一个新任务。

    不直接改原任务的状态——保留失败记录本身有价值（能看出是哪天开始出问题的），
    重跑产出的新记录也让「这次和上次有什么不同」可比较。
    """
    from ..jobs.queue import enqueue

    job = db.session.get(Job, job_id)
    if job is None:
        flash("任务不存在", "error")
        return redirect(url_for("web.jobs_page"))

    new_job = enqueue(job.type, job.params or {}, priority=job.priority, paper_id=job.paper_id)
    flash(f"已重新入队：{new_job.id}", "ok")
    return redirect(url_for("web.jobs_page"))


@web_bp.post("/tasks/scan")
def trigger_scan():
    from ..jobs.queue import enqueue

    job = enqueue("scan", {}, dedupe_key="scan")
    flash(f"扫描任务已入队（{job.id}）", "ok")
    return redirect(request.referrer or url_for("web.dashboard"))


@web_bp.post("/tasks/index")
def trigger_index():
    from ..jobs.queue import enqueue

    job = enqueue("index", {}, dedupe_key="index")
    flash(f"索引任务已入队（{job.id}）", "ok")
    return redirect(request.referrer or url_for("web.dashboard"))


@web_bp.route("/papers/add", methods=["GET", "POST"])
def paper_add():
    """按链接 / arXiv 编号 / 标题添加论文。

    分两步是刻意的：**先解析、让用户确认，再下载入库**。
    标题匹配偶尔会认错（这是 arXiv 搜索的固有特性，CLI 那边也有同样的
    处理原则），而在下载之前让人看一眼「找到的是哪一篇」，比事后发现
    库里混进了别的论文便宜得多。

    解析本身只要一两秒，可以同步做；下载 + 解析正文要几十秒，交给任务。
    """
    from ..services.ingest import IngestError, resolve_arxiv

    query = (request.args.get("q") or request.form.get("q") or "").strip()
    candidate = None
    error = None

    if request.method == "POST" and query:
        try:
            candidate, why = resolve_arxiv(query)
        except IngestError as exc:
            error = str(exc)
        else:
            if candidate is None:
                error = why
        if candidate is not None:
            # 已经入库过就不必再下第二遍。
            #
            # **必须用去掉版本号的编号比对**：arXiv 返回的是 2006.11239v2，
            # 而库里存的是 2006.11239。拿带版本号的去精确匹配永远不相等，
            # 表现是「明明已经在库里了，却又下载了一遍」，产生一条重复记录。
            existing = (
                db.session.query(Paper)
                .filter(
                    db.or_(
                        Paper.arxiv_id == candidate.versionless_id,
                        Paper.arxiv_id == candidate.arxiv_id,
                        Paper.arxiv_id.startswith(candidate.versionless_id + "v"),
                    )
                )
                .first()
            )
            if existing is not None:
                flash(f"这篇已经在库里了：{(existing.title or '')[:50]}", "ok")
                return redirect(url_for("web.paper_detail", paper_id=existing.id))

    return render_template(
        "paper_add.html",
        query=query,
        candidate=candidate,
        error=error,
    )


@web_bp.post("/papers/add/confirm")
def paper_add_confirm():
    """确认下载。真正的下载与解析交给后台任务。"""
    from ..jobs.queue import enqueue

    arxiv_id = (request.form.get("arxiv_id") or "").strip()
    title = (request.form.get("title") or "").strip()
    if not arxiv_id:
        flash("缺少 arXiv 编号", "error")
        return redirect(url_for("web.paper_add"))

    job = enqueue(
        "ingest",
        {"arxiv_id": arxiv_id, "title": title},
        dedupe_key=f"ingest:{arxiv_id}",
    )
    flash(f"已开始下载《{title[:40] or arxiv_id}》（任务 {job.id[:8]}），可在任务页看进度", "ok")
    return redirect(url_for("web.jobs_page"))


@web_bp.post("/papers/<paper_id>/read")
def trigger_read(paper_id: str):
    """对单篇论文跑精读，产出笔记。

    **这个入口之前根本不存在**：CLI 有 ``kb read --paper``、接口有
    ``POST /papers/<id>/read``，唯独网页端没有按钮。于是「AI 精读」——
    这个系统最主要的能力——在界面上够不着，用户只能去开终端。
    论文页的空状态还写着「到任务页让 AI 精读」，而任务页只能取消和重试
    已有任务，开不了新的。
    """
    from ..jobs.queue import enqueue
    from ..services import papers as papers_service

    paper = papers_service.get_paper(paper_id)
    if paper is None:
        flash("论文不存在", "error")
        return redirect(url_for("web.library"))

    # dedupe_key 保证连点两次不会排两个任务；精读要花模型调用，
    # 重复排队既是浪费也会把队列堵住
    job = enqueue("read", {"paper_id": paper_id}, dedupe_key=f"read:{paper_id}",
                  paper_id=paper_id)
    flash(f"已开始为《{(paper.title or '')[:30]}》生成笔记，可在任务页看进度", "ok")
    return redirect(url_for("web.paper_detail", paper_id=paper_id, job=job.id))


@web_bp.post("/tasks/read-missing")
def trigger_read_missing():
    """给所有还没有精读笔记的论文排队。

    批量入口放在概览页：一篇篇点太慢，而且「哪些还没读」正是概览该回答的问题。
    """
    from ..jobs.queue import enqueue
    from ..models import Note

    done = db.session.query(Note.paper_id).filter(Note.kind == "deep_read")
    pending = [
        row[0]
        for row in db.session.query(Paper.id)
        .filter(Paper.deleted_at.is_(None), Paper.id.notin_(done))
        .all()
    ]
    if not pending:
        flash("所有论文都已经有精读笔记了", "ok")
        return redirect(url_for("web.dashboard"))

    job = enqueue("read", {"paper_ids": pending}, dedupe_key="read:missing")
    flash(f"已入队 {len(pending)} 篇论文的精读任务（{job.id}）", "ok")
    return redirect(url_for("web.jobs_page"))


@web_bp.get("/jobs/<job_id>/status")
def job_status(job_id: str):
    """任务状态的轻量 JSON。

    给页面轮询用。完整的事件流在 ``/api/v1/jobs/<id>/events``，那条是 SSE
    且需要 API Key——网页端为了显示一个进度条去配 Key 不值得。
    这里只回几个字段，轮询开销可以忽略。
    """
    job = db.session.get(Job, job_id)
    if job is None:
        return jsonify({"ok": False, "error": "任务不存在"}), 404
    return jsonify(
        {
            "ok": True,
            "status": job.status,
            "progress": round(float(job.progress or 0), 3),
            "message": job.message or "",
            "error": job.error or "",
            "finished": job.status in {"succeeded", "failed", "cancelled"},
        }
    )


@web_bp.get("/about")
def about():
    return render_template(
        "about.html",
        boot_config=current_app.extensions["kb_boot_config"],
        preflight=current_app.extensions.get("kb_preflight"),
        vector_state=current_app.extensions.get("kb_vector_state", {}),
    )


def _wants_json() -> bool:
    """请求方要 JSON 还是页面重定向。

    判断依据是 ``Accept`` 头，不是 ``HX-Request``——网页端从头到尾没用过 HTMX，
    那个头永远不会出现，按它判断等于这个分支永远不会走。
    编辑器用 fetch 发 ``Accept: application/json``，普通表单提交不带这个头，
    于是同一个端点两条路径都成立。

    保留表单路径是刻意的：禁用 JS 时编辑仍然可用。
    """
    return "application/json" in (request.headers.get("Accept") or "")


# --------------------------------------------------------------------------
# 文库
# --------------------------------------------------------------------------


@web_bp.get("/papers")
def library():
    """论文列表：筛选（含标签分面）、排序、列表/卡片两种视图。"""
    from ..services import papers as papers_service

    page = max(1, request.args.get("page", 1, type=int))
    per_page = 30
    query = request.args.get("q", "").strip()

    # 多选标签：AND 语义（同时具备所有选中标签），与 service 层一致。
    # 用 getlist 而不是 get——单选下拉改多选后，用 get 只会拿到第一个。
    tag_ids = [t for t in request.args.getlist("tag") if t][:12]
    sort = request.args.get("sort") or "added"
    if sort not in papers_service.SORT_VALUES:
        sort = "added"

    rows, _cursor, total = papers_service.list_papers(
        query=query or None,
        tag_ids=tag_ids or None,
        year=request.args.get("year", type=int),
        venue=request.args.get("venue") or None,
        reading_status=request.args.get("reading_status") or None,
        ingest_status=request.args.get("ingest_status") or None,
        has_repo=(lambda v: None if v is None else v.lower() in {"1", "true", "yes"})(
            request.args.get("has_code")
        ),
        sort=sort,
        limit=per_page * page,  # 简化分页：一次取到当前页为止
    )
    papers = rows[-(per_page):] if page > 1 else rows[:per_page]

    # 分页链接要带上全部筛选条件，否则翻到第 2 页筛选就没了。
    # 集中在这里构造：散在模板里手写 url_for 参数，加一个筛选项就会漏一处。
    page_args = {
        key: value
        for key, value in (
            ("q", query),
            ("year", request.args.get("year", "")),
            ("venue", request.args.get("venue", "")),
            ("reading_status", request.args.get("reading_status", "")),
            ("ingest_status", request.args.get("ingest_status", "")),
            ("has_code", request.args.get("has_code", "")),
            ("sort", sort if sort != "added" else ""),
        )
        if value
    }
    if tag_ids:
        page_args["tag"] = tag_ids

    return render_template(
        "library.html",
        papers=papers,
        total=total,
        page=page,
        per_page=per_page,
        pages=max(1, (total + per_page - 1) // per_page),
        page_args=page_args,
        # 分面里每个标签的链接要「保留其它筛选、只改标签」，所以单独给一份
        # 不含 tag 的基线，模板在上面加/减当前这一个标签。
        filter_args={k: v for k, v in page_args.items() if k != "tag"},
        options=papers_service.filter_options(),
        facets=papers_service.tag_facets(),
        sort_options=papers_service.SORT_OPTIONS,
        stats=papers_service.stats(),
        filters={
            "q": query,
            "year": request.args.get("year", ""),
            "venue": request.args.get("venue", ""),
            "reading_status": request.args.get("reading_status", ""),
            "ingest_status": request.args.get("ingest_status", ""),
            "has_code": request.args.get("has_code", ""),
            "sort": sort,
            "tag_ids": tag_ids,
        },
    )


@web_bp.get("/papers/<paper_id>")
def paper_detail(paper_id: str):
    """论文详情：元数据、笔记、关联代码。"""
    from ..services import papers as papers_service

    paper = papers_service.get_paper(paper_id, include_deleted=True)
    if paper is None:
        return render_template("error.html", code=404, message="论文不存在"), 404

    notes = (
        db.session.query(Note)
        .filter(Note.paper_id == paper_id)
        .order_by(Note.updated_at.desc())
        .all()
    )

    # 一篇论文可以有多篇笔记（精读、代码对照、自己写的想法…）。
    # ?note=<id> 指定当前要在工作台里编辑哪一篇；没指定时优先精读笔记——
    # 那是这个系统产出的主力，也是最常回看的一篇。
    active_note = None
    wanted = (request.args.get("note") or "").strip()
    if wanted:
        active_note = next((n for n in notes if n.id == wanted), None)
    if active_note is None:
        active_note = next((n for n in notes if n.kind == "deep_read"), None) or (
            notes[0] if notes else None
        )
    chunks = (
        db.session.query(Chunk)
        .filter(Chunk.paper_id == paper_id)
        .order_by(Chunk.ord)
        .limit(200)
        .all()
    )

    all_tags = db.session.query(Tag).order_by(Tag.dimension, Tag.name).all()

    return render_template(
        "paper.html",
        paper=paper,
        notes=notes,
        active_note=active_note,
        # 刚触发过精读时带上任务 id，页面上显示进度条
        job_id=(request.args.get("job") or "").strip() or None,
        chunks=chunks,
        all_tags=all_tags,
        tag_links={link.tag_id: link for link in paper.tag_links},
    )


@web_bp.get("/papers/<paper_id>/file")
def paper_file(paper_id: str):
    """网页端返回 PDF 本体，供内嵌的 PDF.js 阅读器加载。

    为什么不直接链 ``/api/v1/papers/<id>/file``：那条路由挂了
    ``require_scope("read")``，默认鉴权模式是 API Key——浏览器里的
    ``<a>`` 或 PDF.js 发不出 ``Authorization`` 头，打开就是 401。
    网页端本来就跑在同一套 service 上、靠同源与 CSRF 保护，不需要再来一层 Key。

    ``conditional=True`` 让 ``send_file`` 处理 Range 请求，PDF.js 才能
    边下边看、拖动进度条，而不是等整个文件下载完。
    """
    from ..services import papers as papers_service

    paper = papers_service.get_paper(paper_id, include_deleted=True)
    if paper is None:
        return render_template("error.html", code=404, message="论文不存在"), 404

    if not paper.file_path or not Path(paper.file_path).is_file():
        return render_template("error.html", code=410, message="论文文件已不在磁盘上"), 410

    return send_file(
        paper.file_path,
        mimetype="application/pdf",
        download_name=f"{(paper.title or paper.id)[:80]}.pdf",
        conditional=True,
    )


@web_bp.post("/papers/<paper_id>/update")
def paper_update(paper_id: str):
    """保存论文元数据编辑。"""
    from ..services import papers as papers_service

    values = {}
    for key in ("title", "venue", "doi", "arxiv_id", "abstract", "reading_status"):
        if key in request.form:
            values[key] = request.form.get(key) or None
    if "year" in request.form:
        raw = request.form.get("year") or ""
        values["year"] = int(raw) if raw.isdigit() else None
    if "rating" in request.form:
        raw = request.form.get("rating") or ""
        values["rating"] = int(raw) if raw.isdigit() and raw != "0" else None

    try:
        papers_service.update_paper(paper_id, values)
        flash("已保存", "ok")
    except papers_service.PaperError as exc:
        flash(str(exc), "error")

    return redirect(url_for("web.paper_detail", paper_id=paper_id))


@web_bp.post("/papers/<paper_id>/delete")
def paper_delete(paper_id: str):
    """软删除论文。文件和笔记都保留。"""
    from ..services import papers as papers_service

    try:
        papers_service.soft_delete(paper_id)
        flash("已移出文库（文件与笔记仍保留，可恢复）", "ok")
    except papers_service.PaperError as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.library"))


@web_bp.post("/papers/<paper_id>/restore")
def paper_restore(paper_id: str):
    from ..services import papers as papers_service

    try:
        papers_service.restore(paper_id)
        flash("已恢复", "ok")
    except papers_service.PaperError as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.paper_detail", paper_id=paper_id))


@web_bp.get("/duplicates")
def duplicates_page():
    """重复候选：人工确认合并还是保留。"""
    from ..services import papers as papers_service

    return render_template(
        "duplicates.html",
        duplicates=papers_service.list_duplicates("pending"),
        resolved=papers_service.list_duplicates("merged")[:20],
    )


@web_bp.post("/duplicates/<dup_id>/resolve")
def resolve_duplicate_view(dup_id: str):
    from ..services import papers as papers_service

    action = request.form.get("action", "dismiss")
    try:
        papers_service.resolve_duplicate(dup_id, action)
        flash("已处理", "ok")
    except papers_service.PaperError as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.duplicates_page"))


# --------------------------------------------------------------------------
# 笔记
# --------------------------------------------------------------------------


@web_bp.get("/notes")
def notes_page():
    """笔记列表。"""
    from ..services import notes as notes_service
    from ..services import tagging

    query = request.args.get("q", "").strip()
    rows, _cursor, total = notes_service.list_notes(
        query=query or None,
        paper_id=request.args.get("paper_id") or None,
        kind=request.args.get("kind") or None,
        status=request.args.get("status") or None,
        standalone=request.args.get("standalone") == "1",
        limit=100,
    )

    return render_template(
        "notes.html",
        notes=rows,
        total=total,
        stats=notes_service.stats(),
        suggestion_stats=tagging.suggestion_stats(),
        filters={"q": query, "kind": request.args.get("kind", ""),
                 "standalone": request.args.get("standalone", "")},
    )


@web_bp.get("/notes/<note_id>")
def note_detail(note_id: str):
    """笔记编辑器。"""
    from ..services import notes as notes_service
    from ..services.notesync import detect_external_change

    note = notes_service.get_note(note_id)
    if note is None:
        return render_template("error.html", code=404, message="笔记不存在"), 404

    compare_version = request.args.get("compare", type=int)
    diff_text = None
    if compare_version is not None:
        try:
            diff_text = notes_service.diff_revisions(note_id, compare_version)
        except notes_service.NoteError as exc:
            flash(str(exc), "error")
            compare_version = None

    return render_template(
        "note_edit.html",
        note=note,
        paper=db.session.get(Paper, note.paper_id) if note.paper_id else None,
        revisions=notes_service.list_revisions(note_id),
        diff_text=diff_text,
        compare_version=compare_version,
        sync_state=detect_external_change(note),
    )


@web_bp.post("/notes/<note_id>")
def note_save(note_id: str):
    """保存笔记。

    编辑器走 JSON（静默后台保存），普通表单走 flash + 重定向。
    编辑器不能走重定向：保存会把用户从编辑位置弹走，光标和滚动位置都丢。
    """
    from ..services import notes as notes_service

    wants_json = _wants_json()
    values: dict = {}
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        source = payload
    else:
        source = request.form
    for key in ("title", "content_md", "kind", "status"):
        if key in source:
            values[key] = source[key]

    expected = source.get("version")
    if expected is not None:
        try:
            expected = int(expected)
        except (TypeError, ValueError):
            expected = None

    try:
        note = notes_service.update_note(note_id, values, expected_version=expected)
    except notes_service.NoteError as exc:
        if wants_json:
            # 乐观锁冲突要和普通错误区分开：前端据此提示「刷新后重试」，
            # 而不是简单地说「保存失败」（那会让人以为是网络问题而反复重存）
            return jsonify({"ok": False, "error": str(exc)}), 409
        flash(str(exc), "error")
        return redirect(url_for("web.note_detail", note_id=note_id))

    if wants_json:
        return jsonify({
            "ok": True,
            "version": note.version,
            "updated_at": note.updated_at.isoformat() if note.updated_at else None,
            "file_path": note.file_path,
        })
    flash("已保存", "ok")
    return redirect(url_for("web.note_detail", note_id=note.id))


@web_bp.post("/notes")
def note_create():
    """新建笔记。"""
    from ..services import notes as notes_service

    paper_id = request.form.get("paper_id") or None
    try:
        note = notes_service.create_note(
            title=request.form.get("title") or "未命名笔记",
            content_md=request.form.get("content_md") or "",
            paper_id=paper_id,
            kind=request.form.get("kind") or "manual",
        )
    except notes_service.NoteError as exc:
        flash(str(exc), "error")
        return redirect(request.referrer or url_for("web.notes_page"))

    return redirect(url_for("web.note_detail", note_id=note.id))


@web_bp.post("/notes/<note_id>/delete")
def note_delete_view(note_id: str):
    from ..services import notes as notes_service

    remove_file = request.form.get("remove_file") == "1"
    try:
        notes_service.delete_note(note_id, remove_file=remove_file)
        flash("笔记已删除" + ("（含磁盘文件）" if remove_file else "（磁盘文件保留）"), "ok")
    except notes_service.NoteError as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.notes_page"))


@web_bp.post("/notes/<note_id>/restore/<int:version>")
def note_restore_view(note_id: str, version: int):
    from ..services import notes as notes_service

    try:
        notes_service.restore_revision(note_id, version)
        flash(f"已回滚到版本 {version}", "ok")
    except notes_service.NoteError as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.note_detail", note_id=note_id))


@web_bp.post("/notes/sync")
def notes_sync_view():
    """触发与磁盘的同步。"""
    from ..services import notesync

    try:
        result = notesync.sync_all()
        attention = result["conflict"] + result["missing"]
        if attention:
            flash(
                f"同步完成：外部改动 {result['modified']} 篇已导入，"
                f"{result['conflict']} 篇存在冲突，{result['missing']} 篇文件缺失",
                "warning",
            )
        else:
            flash(f"同步完成：共检查 {result['total']} 篇，全部一致", "ok")
    except Exception as exc:
        log.exception("笔记同步失败")
        flash(f"同步失败：{exc}", "error")
    return redirect(request.referrer or url_for("web.notes_page"))


# --------------------------------------------------------------------------
# 标签
# --------------------------------------------------------------------------


@web_bp.get("/review")
def review_page():
    """待核清单：把散在 85 篇笔记里的 ⚠️ 收集起来处理。

    笔记结尾印着「内容为草稿，需人工核对后再采信」，而 AI 自己也标出了不确定处——
    问题是那些标记散在各自的 markdown 文件里，要看就得逐个打开，等于没有。
    这个页面把它们聚起来，并按「谁的问题」分类：我们管线的账和论文自身的疑点
    混在一起，人很快就不看了。
    """
    from ..services import review

    kind = request.args.get("kind") or ""
    show_done = request.args.get("show_done") == "1"

    flags = review.collect(kinds=[kind] if kind in review.KIND_LABELS else None)
    states = review.state_map([f.flag_id for f in flags])
    for flag in flags:
        flag.state = states.get(flag.flag_id, {}).get("state", "")
        flag.comment = states.get(flag.flag_id, {}).get("comment", "")

    pending = [f for f in flags if not f.state]
    return render_template(
        "review.html",
        flags=flags if show_done else pending,
        pending_count=len(pending),
        total_count=len(flags),
        stats=review.summary(flags),
        kind_labels=review.KIND_LABELS,
        kind=kind,
        show_done=show_done,
    )


@web_bp.post("/review/<flag_id>")
def review_mark(flag_id: str):
    """标记/取消标记一条待核项。"""
    from ..services import review

    action = (request.form.get("action") or "done").strip()
    if action == "undo":
        review.unmark(flag_id)
    else:
        review.mark(
            flag_id,
            request.form.get("note_id") or "",
            note_version=request.form.get("note_version", type=int) or 0,
            state=review.STATE_IGNORED if action == "ignore" else review.STATE_DONE,
            comment=request.form.get("comment") or "",
        )
    return redirect(url_for("web.review_page", **(request.args.to_dict() or {})))


@web_bp.get("/tags")
def tags_page():
    """标签词表与 AI 建议队列。"""
    from ..services import tagging

    return render_template(
        "tags.html",
        tree=tagging.tag_tree(),
        dimensions=[
            ("topic", "主题"), ("method", "方法"), ("task", "任务"),
            ("domain", "领域"), ("venue", "发表场所"), ("status", "状态"),
            ("misc", "其它"),
        ],
        suggestions=tagging.list_suggestions("pending", limit=100),
        suggestion_stats=tagging.suggestion_stats(),
        tag_count=len(tagging.list_tags()),
    )


@web_bp.post("/tags")
def tag_create_view():
    from ..services import tagging

    name = (request.form.get("name") or "").strip()
    dimension = request.form.get("dimension") or "misc"
    if not name:
        flash("标签名不能为空", "error")
    else:
        try:
            tagging.get_or_create_tag(name, dimension=dimension)
            flash(f"已添加标签「{name}」", "ok")
        except tagging.TagError as exc:
            flash(str(exc), "error")
    return redirect(url_for("web.tags_page"))


@web_bp.post("/tags/suggestions/<suggestion_id>/resolve")
def resolve_suggestion_view(suggestion_id: str):
    from ..services import tagging

    action = request.form.get("action", "accept")
    try:
        tag = tagging.resolve_suggestion(suggestion_id, action)
        if action == "accept":
            flash(f"已接受，归入标签「{tag.name}」" if tag else "已接受", "ok")
        else:
            flash("已拒绝。这个词以后不会再被建议。", "ok")
    except tagging.TagError as exc:
        flash(str(exc), "error")
    return redirect(url_for("web.tags_page"))


@web_bp.post("/tags/suggestions/accept-all")
def accept_all_suggestions_view():
    from ..services import tagging

    accepted = tagging.accept_all_suggestions()
    flash(f"已接受 {accepted} 条建议", "ok")
    return redirect(url_for("web.tags_page"))


@web_bp.post("/tags/<tag_id>/delete")
def tag_delete_view(tag_id: str):


    tag = db.session.get(Tag, tag_id)
    if tag is None:
        flash("标签不存在", "error")
    else:
        name = tag.name
        db.session.delete(tag)
        db.session.commit()
        flash(f"已删除标签「{name}」", "ok")
    return redirect(url_for("web.tags_page"))


@web_bp.get("/search")
def search_page():
    """检索页。没有查询词时展示索引状态，让用户知道「现在能不能搜」。"""
    from ..services.embedding import embedding_stats
    from ..services.indexer import index_stats

    query = (request.args.get("q") or "").strip()
    mode = request.args.get("mode") or "hybrid"
    hits = []
    elapsed_ms = 0

    if query:
        import time

        from ..services.search import search

        settings = current_app.extensions["kb_settings"]
        started = time.perf_counter()
        hits = search(
            query,
            limit=int(settings.get("retrieval.top_k")),
            mode=mode,
            group_by_paper=bool(settings.get("retrieval.group_by_paper")),
            rrf_k=int(settings.get("retrieval.rrf_k")),
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

    return render_template(
        "search.html",
        query=query,
        mode=mode,
        hits=hits,
        elapsed_ms=elapsed_ms,
        index_stats=index_stats(),
        embedding_stats=embedding_stats(),
    )


@web_bp.post("/papers/upload")
def paper_upload():
    """上传 PDF。"""
    from ..services import papers as papers_service

    file = request.files.get("file")
    if file is None or not file.filename:
        flash("请选择要上传的 PDF 文件", "error")
        return redirect(url_for("web.library"))

    # 上限来自启动级配置：它也同时设着 Flask 的 MAX_CONTENT_LENGTH，
    # 所以超限的请求在进入视图之前就会被 Werkzeug 拒掉，这里是第二道防线。
    cfg = current_app.extensions["kb_boot_config"]
    try:
        paper = papers_service.store_upload(file, max_mb=cfg.max_upload_mb)
        flash(f"已上传：{(paper.title or paper.id)[:60]}", "ok")
        return redirect(url_for("web.paper_detail", paper_id=paper.id))
    except papers_service.PaperError as exc:
        flash(str(exc), "error")
        return redirect(url_for("web.library"))
