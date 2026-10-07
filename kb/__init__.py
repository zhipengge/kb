"""kb —— 论文知识库系统。

应用工厂。启动顺序是刻意安排的，每一步都依赖前一步：

  1. 解析启动级配置（含创建数据目录、装载/生成加密密钥）
  2. 起日志
  3. 创建 Flask app、初始化扩展
  4. 给数据库引擎挂上 PRAGMA 与向量扩展加载
  5. **自检**：日志模式、FTS5、trigram、sqlite-vec —— 结果存进 app.extensions，
     设置页与 /api/v1/system/health 都会展示
  6. 首次启动时把默认设置与初始标签词表播种进库
  7. 注册蓝图与 CLI
  8. 按需启动内嵌 worker

自检刻意排在路由注册之前：能力缺失时应用仍然要能起来（用户得进设置页看原因），
但不能等到第一次检索才以「返回空结果」的方式暴露。
"""

from __future__ import annotations

import logging
import os
import sys

from flask import Flask, render_template, request
from markupsafe import Markup

from .config import BootConfig, load_boot_config
from .extensions import csrf, db, limiter, migrate
from .settings import Settings, make_fernet

__version__ = "0.1.0"

log = logging.getLogger(__name__)


def _configure_logging(cfg: BootConfig) -> None:
    level = getattr(logging, cfg.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # 这几家的 INFO 噪音很大，压到 WARNING
    for noisy in ("httpx", "httpcore", "urllib3", "werkzeug", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def create_app(config_file: str | os.PathLike[str] | None = None) -> Flask:
    cfg = load_boot_config(config_file)
    _configure_logging(cfg)

    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.config.update(cfg.as_flask_config())
    app.config["KB_VERSION"] = __version__

    # 启动级配置对象本身也放进去：数据目录、密钥来源等信息在设置页要用
    app.extensions["kb_boot_config"] = cfg  # type: ignore[assignment]

    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    limiter.init_app(app)

    # **对外接口必须豁免 CSRF**，否则外部 agent 一个请求都发不出去。
    #
    # CSRF 保护的是「浏览器在用户不知情时携带 cookie 凭证发请求」——
    # 它依赖浏览器会自动附带凭证这一点。而接口客户端用的是显式
    # `Authorization: Bearer`，攻击者的页面无法让受害者的浏览器
    # 自动带上这个头，所以这层攻击面本来就不存在。
    #
    # 实测影响：豁免之前 `POST /api/v1/ask` 一律返回 400
    # 「The CSRF token is missing.」——路由全都在、文档也写好了，
    # 但对任何外部调用方都是死的。这类问题在网页端完全测不出来，
    # 因为网页表单都带 csrf_token 字段。
    from .api import api_bp

    csrf.exempt(api_bp)

    # 设置服务（进程内缓存，写入时失效）。自己管理 app context，
    # 因此也能在 CLI 与后台 worker 线程里直接使用。
    app.extensions["kb_settings"] = Settings(  # type: ignore[assignment]
        app, fernet=make_fernet(cfg.secret_key)
    )

    _attach_database(app, cfg)

    from .cli import register_cli

    register_cli(app)

    from .api import api_bp
    from .web import web_bp

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    _register_error_handlers(app)
    _register_context(app)

    if cfg.worker_embedded:
        _start_worker(app)

    log.info(
        "kb %s 已就绪 | 数据目录 %s | 数据库 %s | 密钥来源 %s",
        __version__, cfg.data_dir, cfg.db_path, cfg.secret_key_source,
    )
    return app


def _attach_database(app: Flask, cfg: BootConfig) -> None:
    """挂 PRAGMA / 向量扩展，并跑一次能力自检。"""
    from .sqlite import preflight, register_engine_pragmas, register_vec_loader

    # 自检先用原生 sqlite3 跑一遍（不经过 ORM，事务边界可控）
    report = preflight(cfg.db_path)
    app.extensions["kb_preflight"] = report  # type: ignore[assignment]

    if not report.ok:
        for problem in report.problems:
            log.error("数据库自检：%s", problem)
    for note in report.notes:
        log.warning("数据库自检：%s", note)

    with app.app_context():
        engine = db.engine
        register_engine_pragmas(engine)
        app.extensions["kb_vector_state"] = register_vec_loader(engine)  # type: ignore[assignment]

        from .models import Base

        Base.metadata.create_all(engine)

        # FTS5 虚拟表与同步触发器不在 SQLAlchemy 的 metadata 里（它们不是普通表），
        # 所以单独建。幂等，每次启动都会确保存在。
        from .services.fts import ensure_fts

        app.extensions["kb_fts_state"] = ensure_fts(engine)  # type: ignore[assignment]

    with app.app_context():
        from .services.seed import ensure_seed_data

        ensure_seed_data()


def _start_worker(app: Flask) -> None:
    """启动内嵌的后台任务 worker。

    只在开发/单进程部署下使用。用 gunicorn 多 worker 时必须关掉
    （KB_WORKER_EMBEDDED=0），否则每个 worker 都会跑一份任务循环 ——
    这是这类「数据库当队列」设计最经典的翻车方式。

    **CLI 命令下也要关掉。** 命令行工具（``flask kb import`` 之类）自己就会
    写数据库，而它们做的事情往往带慢 I/O（下载 PDF、解析论文），
    期间事务一直开着。此时 worker 线程在同一个进程里抢写锁，
    会等满 busy_timeout 然后抛 `database is locked`——
    表现为「跑一次批量导入，后台任务全挂了」。
    """
    import os

    if os.environ.get("FLASK_RUN_FROM_CLI"):
        log.debug("检测到 Flask CLI 环境，跳过内嵌 worker（避免与命令本身的写入争锁）")
        return

    from .jobs.queue import start_embedded_worker

    start_embedded_worker(app)


def _register_error_handlers(app: Flask) -> None:
    from .api.envelope import error_response

    class ApiError(Exception):
        """业务异常。带上 HTTP 状态码与机器可读的错误码。"""

        def __init__(self, code: str, message: str, status: int = 400, details: dict | None = None):
            super().__init__(message)
            self.code = code
            self.message = message
            self.status = status
            self.details = details or {}

    app.extensions["kb_api_error"] = ApiError  # type: ignore[assignment]

    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError):
        if request.path.startswith("/api/"):
            return error_response(exc.code, exc.message, exc.status, exc.details)
        return render_template("error.html", code=exc.status, message=exc.message), exc.status

    @app.errorhandler(404)
    def _not_found(_exc):
        if request.path.startswith("/api/"):
            return error_response("not_found", "资源不存在", 404)
        return render_template("error.html", code=404, message="页面不存在"), 404

    @app.errorhandler(500)
    def _server_error(exc):
        log.exception("未处理的异常")
        if request.path.startswith("/api/"):
            return error_response("internal_error", "服务器内部错误", 500)
        return render_template("error.html", code=500, message="服务器内部错误"), 500

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            return exc
        log.exception("未处理的异常：%s", exc)
        if request.path.startswith("/api/"):
            return error_response("internal_error", str(exc), 500)
        return render_template("error.html", code=500, message=str(exc)), 500


def _register_context(app: Flask) -> None:
    """注入所有模板都要用的东西：主题、导航、版本。"""

    @app.context_processor
    def _inject():
        from flask import current_app

        settings: Settings = current_app.extensions["kb_settings"]
        theme, theme_source = settings.get_with_source("appearance.theme")
        return {
            "kb_version": current_app.config.get("KB_VERSION", ""),
            "theme": theme,
            "theme_source": theme_source,
            "accent": settings.get("appearance.accent"),
            "font_size": settings.get("appearance.font_size"),
            "density": settings.get("appearance.density"),
            "radius": settings.get("appearance.radius"),
            "reader_font": settings.get("appearance.reader_font"),
        }

    @app.template_filter("human_size")
    def _human_size(num: float | None) -> str:
        if not num:
            return "—"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if num < 1024:
                return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
            num /= 1024
        return f"{num:.1f} PB"

    @app.template_filter("timeago")
    def _timeago(value) -> str:
        from .utils.time import humanize_delta

        return humanize_delta(value)

    @app.template_filter("markdown")
    def _markdown(value) -> Markup:
        """把 Markdown 渲染成安全的 HTML。

        安全链条有两道：

          1. markdown-it 以 ``html=False`` 运行，原文里的原始 HTML 会被当作
             普通文本转义，而不是执行；
          2. 再用 nh3 过一遍，只放行白名单内的标签与属性。

        两道是刻意的冗余。知识库里的 Markdown 有三个来源——用户手写、
        AI 生成、从外面导入——其中后两者都可能带上你不想要的东西。
        只靠一层的话，任何一次配置疏漏都会变成 XSS。
        """
        import nh3
        from markdown_it import MarkdownIt
        from mdit_py_plugins.footnote import footnote_plugin
        from mdit_py_plugins.tasklists import tasklists_plugin

        if value is None:
            return Markup("")

        md = (
            MarkdownIt("commonmark", {"linkify": True, "typographer": False})
            .enable("table")
            .enable("strikethrough")
            .use(footnote_plugin)
            .use(tasklists_plugin)
        )
        raw = md.render(str(value))

        cleaned = nh3.clean(
            raw,
            tags={
                "p", "br", "hr", "strong", "em", "del", "code", "pre", "blockquote",
                "ul", "ol", "li", "a", "img", "h1", "h2", "h3", "h4", "h5", "h6",
                "table", "thead", "tbody", "tr", "th", "td", "span", "sup", "sub",
                "div", "input", "section",
            },
            attributes={
                "a": {"href", "title"},
                "img": {"src", "alt", "title"},
                # input 用来渲染任务列表的复选框
                "input": {"type", "checked", "disabled"},
                "code": {"class"},
                "span": {"class"},
                "div": {"class"},
            },
            url_schemes={"http", "https", "mailto", "kb"},
        )

        # 已经只含允许的标签与属性。不用 Markup 的话 Jinja 会二次转义，
        # 页面就变成一堆可见的尖括号了。
        return Markup(cleaned)  # noqa: S704

    @app.template_filter("highlight")
    def _highlight(value) -> Markup:
        """把检索片段渲染成带高亮的 HTML。

        顺序很关键：**先转义、再替换标记**。

        片段来自论文原文——也就是用户上传的任意文件。如果先把 \\x02/\\x03
        换成 <mark> 再交给模板，原文里的 <script> 就会原样进入页面。
        先 escape 把尖括号变成实体，此时内容已经无害，再插入我们自己的
        标签才是安全的。
        """
        from markupsafe import Markup, escape

        if value is None:
            return Markup("")
        # 注意不能写成 escape(x).replace("\x02", "<mark>")：
        # markupsafe 出于安全考虑会**转义 replace 的参数**，结果就是
        # 自己的 <mark> 变成 &lt;mark&gt; 显示出来。必须先拿到转义后的
        # 纯文本，再在其上插入我们自己的标签。
        escaped_plain = str(escape(str(value)))
        return Markup(  # noqa: S704 - 内容已在上一步转义，这里插入的是自己写死的标签
            escaped_plain.replace("\x02", "<mark>").replace("\x03", "</mark>")
        )


__all__ = ["__version__", "create_app"]
