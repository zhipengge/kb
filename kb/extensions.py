"""Flask 扩展实例。

扩展对象在这里集中创建、在 app factory 里初始化，避免各处 import 时
产生循环依赖，也让「这个应用用了哪些扩展」一眼可见。
"""

from __future__ import annotations

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

from .models.base import Base

# model_class=Base 让 Flask-SQLAlchemy 用我们自己的声明式基类
db = SQLAlchemy(model_class=Base)

migrate = Migrate()

csrf = CSRFProtect()

limiter = Limiter(
    key_func=get_remote_address,
    # 内存存储：单进程部署下够用，且不引入 Redis。
    # 注意它对多 worker 是「每 worker 各自限流」，真实上限会被放大——
    # 文档里写明了这点，需要严格限流时应改用共享存储。
    storage_uri="memory://",
    default_limits=[],  # 默认不限制，由设置项按需开启
    headers_enabled=True,
)

__all__ = ["csrf", "db", "limiter", "migrate"]
