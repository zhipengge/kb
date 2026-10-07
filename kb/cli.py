"""命令行入口：``flask kb <命令>``。

命令行存在的意义不只是方便：后台任务需要能**脱离 Web 进程**运行
（生产环境用 gunicorn 多 worker 时必须这样），配置需要能被脚本批量改，
密钥需要有地方签发。这些都走 CLI，而不是只留一个网页界面。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click
from flask import Flask, current_app

log = logging.getLogger(__name__)


def register_cli(app: Flask) -> None:
    @app.cli.group("kb")
    def kb_group() -> None:
        """论文知识库管理命令。"""

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    @kb_group.command("selfcheck")
    def selfcheck() -> None:
        """检查运行环境与数据库能力。"""
        cfg = current_app.extensions["kb_boot_config"]
        report = current_app.extensions.get("kb_preflight")
        vec_state = current_app.extensions.get("kb_vector_state", {})

        click.echo("环境")
        click.echo(f"  数据目录    {cfg.data_dir}")
        click.echo(f"  数据库      {cfg.db_path}")
        click.echo(f"  密钥来源    {cfg.secret_key_source}")
        click.echo(f"  监听        {cfg.host}:{cfg.port}")
        click.echo(f"  内嵌 worker {'开' if cfg.worker_embedded else '关'}")

        if report is not None:
            click.echo("\n数据库")
            click.echo(f"  SQLite      {report.sqlite_version}")
            click.echo(f"  日志模式    {report.journal_mode}")
            click.echo(f"  外键约束    {'开' if report.foreign_keys else '关'}")
            click.echo(f"  FTS5        {'可用' if report.fts5 else '不可用'}")
            click.echo(f"  trigram     {'可用' if report.fts5_trigram else '不可用'}")
            click.echo(
                f"  sqlite-vec  {vec_state.get('version') or report.vector_version or '不可用'}"
            )

            if report.problems:
                click.echo("\n问题")
                for problem in report.problems:
                    click.secho(f"  ! {problem}", fg="yellow")
            if report.notes:
                click.echo("\n提示")
                for note in report.notes:
                    click.secho(f"  - {note}", fg="cyan")

            if report.ok:
                click.secho("\n数据库自检通过", fg="green")
            else:
                click.secho("\n数据库自检未通过", fg="red")
                raise SystemExit(1)

    @kb_group.command("info")
    def info() -> None:
        """显示当前配置（密钥以掩码显示）。"""
        from .extensions import db
        from .models import Chunk, Job, Note, Paper
        from .settings import Settings

        settings: Settings = current_app.extensions["kb_settings"]

        click.echo("配置")
        for key, item in sorted(settings.all_with_source().items()):
            mark = "*" if item["source"] == "db" else " "
            click.echo(f" {mark} {key:28} {item['value']!r}")

        click.echo("\n统计")
        for label, model in (
            ("论文", Paper),
            ("笔记", Note),
            ("分块", Chunk),
        ):
            count = db.session.query(model).count()
            click.echo(f"  {label:6} {count}")
        pending = db.session.query(Job).filter(Job.status.in_(("queued", "running"))).count()
        click.echo(f"  进行中任务 {pending}")

        # 索引是不是用当前规则建的。放在 info 里而不是藏进 index 子命令，
        # 因为「检索悄悄变差」这件事只有常看一眼才会被发现。
        from .services.indexer import note_index_report, stale_chunk_report

        notes_report = note_index_report()
        if notes_report["up_to_date"]:
            click.echo(
                f"  笔记索引 {notes_report['indexed_notes']}/{notes_report['notes']} 篇"
                f"（{notes_report['chunks']} 块）"
            )
        else:
            # 笔记没进索引的表现只是「有些东西搜不到」，不会有任何报错，
            # 所以这个数字要在自检里主动冒出来
            click.secho(
                f"  ! 有 {notes_report['missing']} 篇笔记没进索引"
                f"（{notes_report['indexed_notes']}/{notes_report['notes']} 篇已索引）",
                fg="yellow",
            )
            click.echo("    重建：kb index-notes")

        report = stale_chunk_report()
        if report["up_to_date"]:
            click.echo(f"  索引规则 v{report['rules_version']}（全部为当前版本）")
        else:
            click.secho(
                f"  ! 索引规则 v{report['rules_version']}："
                f"{report['stale']:,} 个分块是旧规则建的"
                f"（涉及 {report['stale_papers']} 篇论文）",
                fg="yellow",
            )
            click.echo("    重建：kb index --force --wait")
        click.echo("\n(* 表示已自定义，其余为默认值)")

    # ------------------------------------------------------------------
    # 数据维护
    # ------------------------------------------------------------------
    @kb_group.command("scan")
    @click.option("--root", "roots", multiple=True, help="指定根目录，默认用设置里的全部根目录")
    @click.option("--full", is_flag=True, help="忽略 mtime/size 快路径，强制重新检查所有文件")
    @click.option("--wait", is_flag=True, help="在命令内直接同步执行，不入队")
    def scan(roots: tuple[str, ...], full: bool, wait: bool) -> None:
        """扫描论文目录。"""
        from .jobs.queue import enqueue
        from .services.scanner import scan_roots

        root_list = list(roots) or None

        if wait:
            from .jobs.queue import JobContext
            from .models import Job

            job = Job(type="scan", params={"roots": root_list, "full": full})
            from .extensions import db

            db.session.add(job)
            db.session.commit()
            ctx = JobContext(current_app._get_current_object(), job.id)
            result = scan_roots(ctx, roots=root_list, full=full)
            click.echo(json.dumps(result, ensure_ascii=False, indent=2))
            return

        job = enqueue("scan", {"roots": root_list, "full": full}, dedupe_key="scan")
        click.echo(f"已入队扫描任务 {job.id}")

    @kb_group.command("index")
    @click.option("--wait", is_flag=True, help="同步执行")
    @click.option("--force", is_flag=True, help="忽略缓存，强制重新解析")
    def index(wait: bool, force: bool) -> None:
        """解析 PDF 并建立检索索引。"""
        from .jobs.queue import JobContext, enqueue
        from .services.indexer import index_papers

        if wait:
            from .extensions import db
            from .models import Job

            job = Job(type="index", params={"force": force})
            db.session.add(job)
            db.session.commit()
            ctx = JobContext(current_app._get_current_object(), job.id)
            result = index_papers(ctx, force=force)
            # 代码索引跟着一起做。它是另一条流水线，但用户心智里
            # 「重建索引」就该把能检索的东西都建好——分成两条命令的话，
            # 忘了跑第二条的表现是「代码搜不到」，而且不会有任何提示。
            from .services.code_index import index_code_repos

            result["code"] = index_code_repos(force=force)
            click.echo(json.dumps(result, ensure_ascii=False, indent=2))
            return

        job = enqueue("index", {"force": force}, dedupe_key="index")
        click.echo(f"已入队索引任务 {job.id}")

    @kb_group.command("index-notes")
    def index_notes_cmd() -> None:
        """重建笔记索引。

        平时不需要跑：笔记保存时会自动重建自己那一份。这条命令是给
        「索引和笔记对不上」准备的——比如导入了一批外部笔记、
        或者在笔记索引功能上线之前就已经存在的库。
        """
        from .services.indexer import index_notes, note_index_report

        report = note_index_report()
        click.echo(
            f"当前 {report['indexed_notes']}/{report['notes']} 篇笔记已索引，"
            f"共 {report['chunks']} 块"
        )

        result = index_notes()
        click.secho(
            f"完成：重建 {result['notes']} 篇 -> {result['chunks']} 块", fg="green"
        )

    @kb_group.command("graph")
    @click.option("--build", is_flag=True, help="从笔记重建知识图谱（调模型，按篇计费）")
    @click.option("--limit", type=int, default=0, help="只处理前 N 篇")
    def graph_cmd(build: bool, limit: int) -> None:
        """知识图谱：查看规模，或从笔记重建。"""
        from .services import graph

        if build:
            click.echo("正在从笔记抽取实体与关系…")
            result = graph.build(limit=limit)
            click.secho(
                f"完成：{result['ok']} 篇成功，{result['failed']} 篇失败", fg="green"
            )

        s = graph.stats()
        click.echo()
        click.echo(f"  实体      {s['entities']:,}")
        click.echo(f"  关系      {s['relations']:,}（其中带证据 {s['with_evidence']:,}）")
        click.echo(f"  论文关联  {s['paper_links']:,}")

        top = graph.top_entities(limit=12)
        if top:
            click.echo()
            click.secho("  被最多论文提到的实体", bold=True)
            for item in top:
                click.echo(f"    {item['papers']:3} 篇  [{item['type']:8}] {item['name'][:44]}")

    @kb_group.command("web-cache")
    @click.option("--clear", is_flag=True, help="清空缓存（默认只显示状态）")
    def web_cache(clear: bool) -> None:
        """查看或清空联网检索缓存。"""
        # 过期判断交给服务层，不要在 CLI 里写 SQL 比较时间——
        # SQLite 存的是 naive UTC，拿 utcnow() 去 filter 会因时区问题算错
        from .services.websearch import cache_stats, clear_cache

        stats = cache_stats()
        click.echo(
            f"联网检索缓存：{stats['total']} 条"
            f"（未过期 {stats['alive']}，已过期 {stats['expired']}）"
        )
        if not stats["total"]:
            click.echo("  （还没有缓存。联网检索一次之后就会有。）")

        if clear:
            removed = clear_cache()
            click.secho(f"已清空 {removed} 条", fg="green")

    # ------------------------------------------------------------------
    # 任务
    # ------------------------------------------------------------------
    @kb_group.command("worker")
    @click.option("--concurrency", type=int, default=None, help="并发线程数，默认用设置里的值")
    @click.option("--once", is_flag=True, help="只跑一轮就退出（便于脚本化）")
    def worker(concurrency: int | None, once: bool) -> None:
        """独立运行后台任务 worker。

        生产环境推荐这样跑，而不是依赖 Web 进程内嵌的 worker——
        gunicorn 起多个 worker 时，每个进程都会跑一份任务循环。
        """
        from .jobs.queue import Worker

        settings = current_app.extensions["kb_settings"]
        n = concurrency or int(settings.get("jobs.concurrency"))
        w = Worker(current_app._get_current_object(), concurrency=n)

        if once:
            from .extensions import db
            from .jobs.queue import claim_next, recover_stale_jobs
            from .jobs.tasks import TASKS

            with current_app.app_context():
                recover_stale_jobs(int(settings.get("jobs.lease_seconds")))
                job = claim_next(f"cli:{__import__('os').getpid()}")
                if job is None:
                    click.echo("队列为空")
                    return
                click.echo(f"执行任务 {job.id} ({job.type})")
                w._run_one(job, TASKS)
                db.session.expire_all()
                from .models import Job

                fresh = db.session.get(Job, job.id)
                click.echo(f"结果：{fresh.status}")
                if fresh.error:
                    click.echo(f"错误：{fresh.error}")
            return

        w.start()
        click.echo(f"worker 已启动（{n} 线程，Ctrl-C 退出）")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("\n正在停止…")
            w.stop()

    @kb_group.command("jobs")
    @click.option("--limit", type=int, default=20)
    def jobs_list(limit: int) -> None:
        """列出最近的任务。"""
        from .extensions import db
        from .models import Job

        rows = db.session.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
        if not rows:
            click.echo("暂无任务")
            return
        for job in rows:
            click.echo(
                f"{job.id}  {job.status:10} {job.type:10} "
                f"{job.progress * 100:5.1f}%  {job.message or ''}"
            )
            if job.error:
                click.secho(f"    错误：{job.error}", fg="red")

    @kb_group.command("import")
    @click.option("--list", "list_file", type=click.Path(exists=True), required=True,
                  help="每行一个标题或 arXiv 编号的文本文件")
    @click.option("--limit", type=int, default=0, help="只处理前 N 条（调试用）")
    @click.option("--dry-run", is_flag=True, help="只解析，不下载")
    @click.option("--min-similarity", type=float, default=78.0, help="标题匹配阈值")
    def import_papers(list_file: str, limit: int, dry_run: bool, min_similarity: float) -> None:
        """按标题批量从 arXiv 入库。

        对每一行：解析成 arXiv 编号 -> 下载 PDF -> 建立记录。
        解析不出来或相似度不够的会列在最后，**不会瞎猜**——
        猜错的代价是把两篇不同的论文混成一条记录，而用户不会察觉。
        """
        from .extensions import db
        from .services import papers as papers_service
        from .services.ingest import IngestError, download_pdf, resolve_arxiv
        from .services.paths import ensure_default_dirs

        settings = current_app.extensions["kb_settings"]
        roots = settings.papers_roots
        if not roots:
            click.secho("没有配置论文根目录", fg="red")
            raise SystemExit(1)

        target_dir = Path(roots[0]) / "_arxiv"
        _created, errors = ensure_default_dirs(settings)
        for message in errors:
            click.secho(f"  ! {message}", fg="yellow")
        target_dir.mkdir(parents=True, exist_ok=True)

        lines = [
            line.strip()
            for line in Path(list_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if limit:
            lines = lines[:limit]

        click.echo(f"共 {len(lines)} 条，目标目录 {target_dir}\n")

        resolved: list[tuple[str, str]] = []
        failed: list[tuple[str, str]] = []

        for index, raw in enumerate(lines, 1):
            try:
                candidate, why = resolve_arxiv(raw, min_similarity=min_similarity)
            except IngestError as exc:
                failed.append((raw, str(exc)))
                click.secho(f"[{index:3}/{len(lines)}] ✗ {raw[:46]} — {exc}", fg="yellow")
                continue

            if candidate is None:
                failed.append((raw, why))
                click.secho(f"[{index:3}/{len(lines)}] ✗ {raw[:46]} — {why[:60]}", fg="yellow")
                continue

            click.echo(
                f"[{index:3}/{len(lines)}] ✓ {candidate.arxiv_id:14} {candidate.title[:56]}"
            )
            resolved.append((raw, candidate.arxiv_id))

            if dry_run:
                continue

            path, message = download_pdf(
                candidate.arxiv_id, target_dir,
                filename=f"{candidate.arxiv_id.replace('.', '_')}_{candidate.title[:60]}.pdf",
            )
            if path is None:
                failed.append((raw, message))
                click.secho(f"           下载失败：{message}", fg="red")
                continue

            try:
                paper = papers_service.create_from_path(
                    str(path), source="arxiv", title=candidate.title
                )
                # 补上检索到的元数据——比从 PDF 里抽的准确
                if candidate.authors and not paper.authors:
                    papers_service.update_paper(paper.id, {"authors": candidate.authors})
                db.session.commit()
            except Exception as exc:
                failed.append((raw, f"入库失败：{exc}"))
                click.secho(f"           入库失败：{exc}", fg="red")

        click.echo()
        click.secho(f"完成：解析成功 {len(resolved)}，失败 {len(failed)}", bold=True)
        if failed:
            click.secho("\n未处理的条目（需要人工确认）：", fg="yellow")
            for raw, reason in failed:
                click.echo(f"  · {raw[:60]}")
                click.echo(f"    {reason[:100]}")

    @kb_group.command("code")
    @click.option("--all", "process_all", is_flag=True, help="处理所有还没有关联代码的论文")
    @click.option("--paper", "paper_id", default=None, help="只处理指定论文")
    @click.option("--find-only", is_flag=True, help="只查找并记录，不克隆")
    @click.option("--limit", type=int, default=0, help="最多处理 N 篇")
    def fetch_code(process_all: bool, paper_id: str | None, find_only: bool, limit: int) -> None:
        """为论文查找并克隆开源代码。

        先看论文正文里有没有作者给的项目链接，没有再按标题搜 GitHub。
        搜索会排除「复现」「笔记」类仓库——把复现仓库当成官方实现挂上去，
        比没有代码更糟。
        """
        from .extensions import db
        from .models import CodeRepo, Paper
        from .services import coderepo
        from .services.paths import ensure_default_dirs

        settings = current_app.extensions["kb_settings"]
        codes_root = settings.codes_root
        if not codes_root:
            click.secho("没有配置代码目录", fg="red")
            raise SystemExit(1)
        _created, errors = ensure_default_dirs(settings)
        for message in errors:
            click.secho(f"  ! {message}", fg="yellow")

        query = db.session.query(Paper).filter(Paper.deleted_at.is_(None))
        if paper_id:
            query = query.filter(Paper.id == paper_id)
        elif process_all:
            linked = db.session.query(CodeRepo.paper_id).filter(CodeRepo.paper_id.isnot(None))
            query = query.filter(Paper.id.notin_(linked))
        else:
            click.secho("需要指定 --all 或 --paper", fg="yellow")
            raise SystemExit(1)

        papers = query.order_by(Paper.id).all()
        if limit:
            papers = papers[:limit]

        click.echo(f"共 {len(papers)} 篇待处理\n")

        found = cloned = skipped = 0
        for index, paper in enumerate(papers, 1):
            label = (paper.title or paper.id)[:46]
            try:
                candidate, why = coderepo.find_repo(paper)
            except Exception as exc:
                click.secho(f"[{index:3}/{len(papers)}] ✗ {label} — {exc}", fg="yellow")
                continue

            if candidate is None:
                skipped += 1
                click.echo(f"[{index:3}/{len(papers)}] – {label} — {why[:60]}")
                continue

            found += 1
            click.secho(f"[{index:3}/{len(papers)}] ✓ {label}", fg="green")
            click.echo(f"          {candidate.full_name}  ★{candidate.stars}  ← {why[:60]}")

            local_path = None
            if not find_only:
                path, message = coderepo.clone_repo(
                    candidate.url, Path(codes_root),
                    name=f"{paper.id[:8]}-{candidate.full_name.replace('/', '__')}",
                )
                if path is None:
                    click.secho(f"          克隆失败：{message}", fg="red")
                else:
                    local_path = str(path)
                    cloned += 1
                    click.echo(f"          {message}")

            repo = CodeRepo(
                paper_id=paper.id,
                name=candidate.full_name,
                url=candidate.url,
                local_path=local_path,
                mapping_source=candidate.source,
                mapping_confidence=candidate.confidence,
                mapping_evidence=candidate.evidence,
            )
            if local_path:
                repo.language_stats = coderepo.repo_size(Path(local_path))
                repo.readme_excerpt = coderepo.read_repo_readme(Path(local_path), limit=2000)
            db.session.add(repo)
            db.session.commit()

        click.echo()
        click.secho(f"完成：找到 {found}，克隆 {cloned}，未找到 {skipped}", bold=True)

    @kb_group.command("read")
    @click.option("--all", "process_all", is_flag=True, help="处理所有已索引但还没生成笔记的论文")
    @click.option("--paper", "paper_id", default=None, help="只处理指定论文")
    @click.option("--limit", type=int, default=0, help="最多处理 N 篇")
    @click.option("--force", is_flag=True, help="已经有笔记的论文也重做")
    @click.option("--no-stage-cache", is_flag=True,
                  help="连精读/打标的中间产物也重新生成（改了阶段实现但没升版本号时用）")
    @click.option("--dry-run", is_flag=True, help="只列出会处理哪些论文")
    def read_papers(process_all: bool, paper_id: str | None, limit: int,
                    force: bool, no_stage_cache: bool, dry_run: bool) -> None:
        """深度阅读论文并生成笔记。

        LLM 调用是按篇计费的，所以默认跳过已有精读笔记的论文（``--force`` 覆盖）。
        处理进度会实时打印，中途中断可以重跑——已完成的阶段会被缓存复用。
        """
        from .extensions import db
        from .models import Note, Paper
        from .services.indexer import index_paper
        from .services.llm import LLMError, is_configured
        from .services.reading import run_pipeline

        if not is_configured():
            click.secho("模型还没配置好。先跑：flask kb llm import-claude-config", fg="red")
            raise SystemExit(1)

        query = db.session.query(Paper).filter(Paper.deleted_at.is_(None))
        if paper_id:
            query = query.filter(Paper.id == paper_id)
        elif process_all:
            # --force 管的是**哪些论文要处理**：默认跳过已有精读笔记的，
            # 加上它才是全量重做。
            #
            # 它不负责让改过的提示词生效——提示词版本参与缓存指纹，
            # 改了提示词或 schema，旧产物会自然失效并重新生成。
            # 这两件事原先共用一个 flag，结果是「只改了渲染层」也要把
            # 全部论文的 LLM 调用重烧一遍，而重渲染本来是可以走缓存的。
            if not force:
                done = db.session.query(Note.paper_id).filter(Note.kind == "deep_read")
                query = query.filter(Paper.id.notin_(done))
        else:
            click.secho("需要指定 --all 或 --paper", fg="yellow")
            raise SystemExit(1)

        papers = query.order_by(Paper.id).all()
        if limit:
            papers = papers[:limit]

        click.echo(f"共 {len(papers)} 篇待精读")
        if dry_run:
            for paper in papers:
                click.echo(f"  · {(paper.title or paper.id)[:70]}")
            return
        click.echo()

        ok = failed = 0
        total_tokens = 0
        for index, paper in enumerate(papers, 1):
            label = (paper.title or paper.id)[:52]
            click.echo(f"[{index:3}/{len(papers)}] {label}")
            try:
                if not db.session.query(Paper).get(paper.id).chunks:
                    click.echo("          索引中…")
                    index_paper(paper)

                result = run_pipeline(paper, force=no_stage_cache)
                if result.errors:
                    failed += 1
                    click.secho(f"          ✗ {'；'.join(result.errors)[:80]}", fg="red")
                    continue

                ok += 1
                total_tokens += result.tokens_used
                stages = "+".join(result.stages_run) or "全部复用缓存"
                click.secho(
                    f"          ✓ {stages}  {result.elapsed_ms / 1000:.0f}s  "
                    f"标签 {len(result.tags_created)}",
                    fg="green",
                )
            except LLMError as exc:
                failed += 1
                click.secho(f"          ✗ 模型调用失败：{exc}", fg="red")
            except Exception as exc:
                failed += 1
                click.secho(f"          ✗ {type(exc).__name__}: {exc}"[:110], fg="red")
                log.exception("精读 %s 失败", paper.id)

        click.echo()
        click.secho(f"完成：成功 {ok}，失败 {failed}", bold=True)
        if total_tokens:
            click.echo(f"  累计消耗 {total_tokens:,} tokens")

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------
    @kb_group.group("llm")
    def llm_group() -> None:
        """模型配置与连通性检查。"""

    @llm_group.command("status")
    def llm_status() -> None:
        """显示当前模型配置（密钥掩码）。"""
        from .services.llm import configuration_status

        status = configuration_status()
        click.echo(f"  服务商      {status['provider']}")
        click.echo(f"  端点        {status['base_url']}")
        click.echo(f"  深度模型    {status['deep_model']}")
        click.echo(f"  轻量模型    {status['fast_model']}")
        click.echo(f"  凭据        {'已配置' if status['has_credential'] else '未配置'}")
        click.echo(f"  输出上限    {status['max_tokens']} tokens")
        if status["ready"]:
            click.secho("\n  配置完整", fg="green")
        else:
            click.secho("\n  配置不完整：", fg="yellow")
            for problem in status["problems"]:
                click.echo(f"    · {problem}")

    @llm_group.command("import-claude-config")
    @click.option("--path", default=None, help="配置文件路径，默认 ~/.claude.json")
    @click.option("--dry-run", is_flag=True, help="只预览，不写入")
    def llm_import(path: str | None, dry_run: bool) -> None:
        """从 Claude Code 的配置里导入模型设置。

        会用 ~/.claude.json 里 env 段的 ANTHROPIC_* 配置，
        包括其中的 API Key。
        """
        from .services.llm import bootstrap

        settings = current_app.extensions["kb_settings"]

        try:
            if dry_run:
                result = bootstrap.preview(path)
                click.secho("将要写入（预览）：", bold=True)
            else:
                result = bootstrap.import_from_claude_config(settings, path=path)
                click.secho("已导入：", bold=True, fg="green")
        except bootstrap.ClaudeConfigError as exc:
            click.secho(f"导入失败：{exc}", fg="red")
            raise SystemExit(1) from exc

        for key, value in result.items():
            click.echo(f"  {key:14} {value}")
        if not dry_run:
            click.secho("\n到「设置 → 模型」可以看到结果。", fg="bright_black")

    @llm_group.command("capabilities")
    @click.option("--probe", is_flag=True, help="实际调用一次，验证能力（会消耗额度）")
    def llm_capabilities(probe: bool) -> None:
        """显示（或实测）当前服务商支持的能力。

        不探测时只展示按端点推断的默认值——那些值是保守的，
        不代表端点真的不支持，只是「不假设它支持」。
        """
        from .services.llm import get_provider
        from .services.llm.probe import probe_capabilities

        try:
            provider = get_provider()
        except Exception as exc:
            click.secho(f"无法构造 Provider：{exc}", fg="red")
            raise SystemExit(1) from exc

        if probe:
            click.echo("正在实测（会消耗少量额度）…\n")
            report = probe_capabilities(provider)
            caps = report["capabilities"]
            if not report.get("fatal"):
                from .services.llm import save_capabilities

                save_capabilities(current_app.extensions["kb_settings"], provider, caps)
                click.secho("结果已保存，后续调用会直接复用（不必重复探测）。\n", fg="bright_black")
        else:
            caps = provider.capabilities.to_dict()

        click.secho(f"模型 {provider.model}", bold=True)
        for key in ("pdf_native", "vision", "prompt_cache", "structured_output",
                    "tool_use", "thinking_control", "streaming"):
            value = caps.get(key)
            mark = click.style("支持", fg="green") if value else click.style("不支持", fg="yellow")
            click.echo(f"  {key:20} {mark}")
        click.echo(f"  {'max_output_tokens':20} {caps.get('max_output_tokens')}")

        for note in caps.get("notes") or []:
            click.secho(f"\n  · {note}", fg="bright_black")
        for problem in report.get("errors", []) if probe else []:
            click.secho(f"\n  ! {problem}", fg="yellow")

    @llm_group.command("test")
    def llm_test() -> None:
        """发一条最小请求，验证配置可用。"""
        from .services.llm import get_provider

        try:
            provider = get_provider()
            response = provider.complete(
                [{"role": "user", "content": "只回复两个字：收到"}], max_tokens=2000
            )
        except Exception as exc:
            click.secho(f"调用失败：{exc}", fg="red")
            raise SystemExit(1) from exc

        click.secho("调用成功", fg="green")
        click.echo(f"  模型      {response.model}")
        click.echo(f"  回复      {response.text[:80]!r}")
        if response.thinking:
            click.echo(f"  思考      {len(response.thinking)} 字符")
        click.echo(
            f"  用量      {response.usage.input_tokens} in / "
            f"{response.usage.output_tokens} out"
        )
        if response.usage.cache_read_tokens == 0 and provider.capabilities.prompt_cache:
            click.secho("  ! 声明支持缓存但本次缓存命中为 0，可能并未真正生效", fg="yellow")

    @llm_group.command("budget")
    @click.option("--days", type=int, default=None, help="只看最近 N 天；默认统计全部历史")
    @click.option("--top", type=int, default=10, help="列出花费最多的 N 篇论文/会话")
    def llm_budget(days: int | None, top: int) -> None:
        """查看模型用量账本。

        显示两个口径：**历史总计**（一共花了多少）和**滚动窗口**
        （闸门看的那 24 小时）。两者不是一回事，混在一起会让人
        以为「明明没超怎么被拦了」。
        """
        from .services import budget

        data = budget.summary(days=days)

        label = f"最近 {days} 天" if days else "全部历史"
        click.secho(f"\n[{label}]", bold=True)
        click.echo(f"  调用      {data['calls']:,} 次")
        click.echo(f"  token     {data['input_tokens']:,} in / {data['output_tokens']:,} out")
        if data["cache_read_tokens"]:
            click.echo(f"  缓存命中  {data['cache_read_tokens']:,}")
        if data["priced"]:
            click.echo(f"  折合      ${data['cost_usd']:.4f}")

        if data["by_kind"]:
            click.echo()
            click.secho("  按环节", bold=True)
            # 按总量降序——最贵的排最上面，这才是「花在哪了」的答案
            ordered = sorted(
                data["by_kind"].items(),
                key=lambda kv: kv[1]["input"] + kv[1]["output"],
                reverse=True,
            )
            for kind, bucket in ordered:
                tokens = bucket["input"] + bucket["output"]
                click.echo(
                    f"    {kind:<10} {tokens:>12,} token  ({bucket['calls']} 次)"
                )

        if top:
            consumers = budget.top_consumers(limit=top)
            if consumers:
                click.echo()
                click.secho("  最贵的对象", bold=True)
                for item in consumers:
                    click.echo(f"    {item['ref']}  {item['tokens']:,} token")

        # ---- 闸门 ----
        click.echo()
        click.secho(f"  闸门（滚动 {data['window_hours']} 小时）", bold=True)
        window = data["window_tokens"]
        if data["limit_tokens"]:
            remaining = data["remaining_tokens"]
            used_pct = window / data["limit_tokens"] * 100 if data["limit_tokens"] else 0
            colour = "red" if data["blocked"] else ("yellow" if used_pct > 80 else "green")
            click.secho(
                f"    已用 {window:,} / {data['limit_tokens']:,} "
                f"({used_pct:.1f}%，剩余 {remaining:,})",
                fg=colour,
            )
        else:
            click.echo(f"    token 上限未设（已用 {window:,}），闸门关闭")
        if data["limit_usd"]:
            if data["priced"]:
                click.echo(
                    f"    金额 ${data['window_cost_usd']:.4f} / ${data['limit_usd']:.2f}"
                )
            else:
                click.secho(
                    "    设了金额上限但没填单价，这道闸不会生效", fg="yellow"
                )
        if data["blocked_calls"]:
            click.secho(f"    被拦下 {data['blocked_calls']} 次", fg="red")
        elif data["blocked_calls_total"]:
            click.echo(
                f"    本窗口未被拦；历史上共拦下 {data['blocked_calls_total']} 次"
            )
        click.echo()

    # ------------------------------------------------------------------
    # 密钥
    # ------------------------------------------------------------------
    @kb_group.group("key")
    def key_group() -> None:
        """管理对外接口的 API Key。"""

    @key_group.command("create")
    @click.option("--name", required=True, help="用途说明，便于日后识别")
    @click.option(
        "--scope", "scopes", multiple=True,
        type=click.Choice(["read", "write", "ingest", "admin"]),
        help="可重复。默认 read。",
    )
    def key_create(name: str, scopes: tuple[str, ...]) -> None:
        """签发一个新的 API Key。明文只显示这一次。"""
        from .services.apikeys import create_key

        scope_list = list(scopes) or ["read"]
        plaintext, row = create_key(name, scope_list)
        click.secho("\n请立即保存这个 Key —— 它不会再次显示：\n", fg="yellow")
        click.secho(f"  {plaintext}\n", fg="green", bold=True)
        click.echo(f"  名称   {row.name}")
        click.echo(f"  前缀   {row.prefix}")
        click.echo(f"  权限   {', '.join(scope_list)}")

    @key_group.command("list")
    def key_list() -> None:
        """列出所有 API Key。"""
        from .extensions import db
        from .models import ApiKey

        rows = db.session.query(ApiKey).order_by(ApiKey.created_at.desc()).all()
        if not rows:
            click.echo("暂无 API Key")
            return
        for row in rows:
            state = "有效" if row.is_active else "已失效"
            click.echo(
                f"{row.prefix}…  {row.name:20} [{', '.join(row.scopes or [])}] "
                f"{state}  调用 {row.call_count} 次"
            )

    @key_group.command("revoke")
    @click.argument("prefix")
    def key_revoke(prefix: str) -> None:
        """按前缀吊销 API Key。"""
        from .extensions import db
        from .models import ApiKey
        from .models.base import utcnow

        rows = db.session.query(ApiKey).filter(ApiKey.prefix == prefix).all()
        if not rows:
            click.secho(f"没有找到前缀为 {prefix} 的 Key", fg="red")
            raise SystemExit(1)
        for row in rows:
            row.revoked_at = utcnow()
        db.session.commit()
        click.secho(f"已吊销 {len(rows)} 个 Key", fg="green")

    # ------------------------------------------------------------------
    # 设置
    # ------------------------------------------------------------------
    @kb_group.group("settings")
    def settings_group() -> None:
        """读写运行时设置。"""

    @settings_group.command("list")
    def settings_list() -> None:
        from .settings import grouped_definitions

        settings = current_app.extensions["kb_settings"]
        values = settings.all_with_source()
        for group, defs in grouped_definitions().items():
            click.secho(f"\n[{group}]", bold=True)
            for definition in defs:
                item = values[definition.key]
                mark = "*" if item["source"] == "db" else " "
                click.echo(f" {mark} {definition.key:28} {item['value']!r}")
                if definition.help:
                    click.secho(f"      {definition.help}", fg="bright_black")

    @settings_group.command("set")
    @click.argument("key")
    @click.argument("value")
    def settings_set(key: str, value: str) -> None:
        """修改一个设置项。"""
        from .settings import SettingsError

        settings = current_app.extensions["kb_settings"]
        try:
            settings.set(key, value, updated_by="cli")
        except SettingsError as exc:
            click.secho(f"设置失败：{exc}", fg="red")
            raise SystemExit(1) from exc
        click.secho(f"{key} = {settings.get(key)!r}", fg="green")

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    @kb_group.command("backup")
    @click.argument("target", type=click.Path())
    def backup(target: str) -> None:
        """把数据库备份到指定文件（不依赖 sqlite3 命令）。"""
        import os

        from .sqlite import vacuum_into

        cfg = current_app.extensions["kb_boot_config"]
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        vacuum_into(cfg.db_path, target)
        size = os.path.getsize(target)
        click.secho(f"已备份到 {target}（{size / 1024 / 1024:.1f} MB）", fg="green")

    @kb_group.command("routes")
    def routes() -> None:
        """列出所有路由（排查 404 时很有用）。"""
        for rule in sorted(current_app.url_map.iter_rules(), key=lambda r: str(r)):
            methods = ",".join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
            click.echo(f"{methods:18} {rule}")


__all__ = ["register_cli"]
