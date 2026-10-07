"""生产入口（gunicorn）。

    pipenv run gunicorn -w 1 --threads 8 -b 0.0.0.0:5000 wsgi:app

**为什么是 -w 1**：后台 worker 内嵌在 Web 进程里，多起一个进程就会多跑一份
任务队列，同一个任务可能被两个进程同时执行。用线程数来撑并发即可
（本应用是 IO 密集：解析 PDF、调用模型、读写磁盘）。

要真正横向扩展时，把 KB_WORKER_EMBEDDED 设为 0，另起 `flask kb worker`
进程专门跑任务，那时就可以随便加 gunicorn worker 了。
"""

from __future__ import annotations

from kb import create_app

app = create_app()
