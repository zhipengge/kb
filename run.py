#!/usr/bin/env python
"""开发服务器入口。

    pipenv run python run.py

生产环境不要用这个（Flask 自带的服务器是单线程的，且不适合暴露在外），
用 wsgi.py 配合 gunicorn。
"""

from __future__ import annotations

from kb import create_app

app = create_app()

if __name__ == "__main__":
    cfg = app.extensions["kb_boot_config"]
    # host/port 来自启动级配置（.env 或 config.toml），不是硬编码
    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug, threaded=True)
