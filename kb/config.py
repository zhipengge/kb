"""启动级配置。

配置分两层，边界是「数据库能不能打开」：

  1. **启动级**（本模块）：数据库放哪、密钥是什么、监听哪个端口。
     来自 代码默认值 < config.toml < 环境变量/.env。数据库自己决定不了数据库在哪，
     所以这些必须在打开数据库之前就确定。

  2. **运行级**（``kb.settings.Settings``）：论文根目录、模型、检索参数、主题外观。
     存在数据库里，网页端可改。数据库一打开就能读到。

刻意**不**把运行级配置回填进 ``app.config``：那会让全局可变状态到处流动，
也会丢掉「这个值现在到底来自哪一层」的信息。设置页需要给每个字段标注来源，
独立的 Settings 服务天然做得到。
"""

from __future__ import annotations

import os
import secrets
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 默认值
# --------------------------------------------------------------------------

# 数据目录的兜底值。**这里必须指向数据真正所在的位置。**
#
# 原先是 "~/.local/share/kb"（用户家目录），2026-10-07 数据迁到了挂载的 SSD 上。
# 兜底值不跟着改的后果很隐蔽：`.env` 一旦丢失（重新 clone 工作区、换机器部署），
# 应用会安静地在旧路径**新建一个空库**——界面正常打开、没有任何报错，
# 只是论文一篇都没有。用户看到的现象是「我的数据全没了」，
# 而真相是它换了个库在跑，真正那份还好端端放在 SSD 上。
#
# 顺带一提，空库意味着 secret.key 也会重新生成，已有的 API Key 一并失效。
#
# 仍然优先读 KB_DATA_DIR 环境变量（见 .env），这个值只是它缺席时的退路。
DEFAULT_DATA_DIR = "/root/kb/data"
DEFAULT_DB_NAME = "kb.sqlite3"

# 用户数据的默认位置。这些是「种子值」——首次启动时写进数据库的 settings 表，
# 之后就以数据库里的值为准（网页端可改）。
DEFAULT_PAPERS_ROOT = "/mnt/papers"
DEFAULT_NOTES_ROOT = "/mnt/kb-notes"
DEFAULT_CODES_ROOT = "/mnt/kb-codes"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


class ConfigError(RuntimeError):
    """启动级配置有问题——这类问题必须在启动时就炸掉，不能拖到运行期。"""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise ConfigError(f"{name} 需要是布尔值（1/0、true/false、yes/no、on/off），收到的是 {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 需要是整数，收到的是 {raw!r}") from exc


def _expand(path: str | os.PathLike[str]) -> Path:
    """展开 ~ 与环境变量，转成绝对路径。"""
    return Path(os.path.expandvars(str(path))).expanduser()


# --------------------------------------------------------------------------
# config.toml（可选）
# --------------------------------------------------------------------------

_TOML_KEYS = {
    "data_dir": "data_dir",
    "db_path": "db_path",
    "host": "host",
    "port": "port",
    "debug": "debug",
    "log_level": "log_level",
    "worker_embedded": "worker_embedded",
    "auth_mode": "auth_mode",
    "max_upload_mb": "max_upload_mb",
    "papers_root": "papers_root",
    "notes_root": "notes_root",
    "codes_root": "codes_root",
}


def _load_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"读取 {path} 失败：{exc}") from exc

    # 允许 [kb] 小节，也允许顶层直接写
    if isinstance(raw.get("kb"), dict):
        raw = {**raw, **raw["kb"]}
    unknown = set(raw) - set(_TOML_KEYS) - {"kb"}
    if unknown:
        raise ConfigError(
            f"{path} 里有无法识别的配置项：{', '.join(sorted(unknown))}。"
            f"可用项：{', '.join(sorted(_TOML_KEYS))}"
        )
    return raw


# --------------------------------------------------------------------------
# 解析结果
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BootConfig:
    """在打开数据库之前就必须确定下来的配置。"""

    data_dir: Path
    db_path: Path
    host: str
    port: int
    debug: bool
    log_level: str
    worker_embedded: bool
    auth_mode: str
    max_upload_mb: int
    secret_key: bytes
    secret_key_source: str

    # 这些会作为种子写进数据库；之后由数据库中的值接管。
    seed_papers_root: str = DEFAULT_PAPERS_ROOT
    seed_notes_root: str = DEFAULT_NOTES_ROOT
    seed_codes_root: str = DEFAULT_CODES_ROOT

    # 派生目录
    uploads_dir: Path = field(default=None)  # type: ignore[assignment]
    artifacts_dir: Path = field(default=None)  # type: ignore[assignment]
    logs_dir: Path = field(default=None)  # type: ignore[assignment]
    cache_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # frozen dataclass 里设置派生字段
        object.__setattr__(self, "uploads_dir", self.data_dir / "uploads")
        object.__setattr__(self, "artifacts_dir", self.data_dir / "artifacts")
        object.__setattr__(self, "logs_dir", self.data_dir / "logs")
        object.__setattr__(self, "cache_dir", self.data_dir / "cache")

    @property
    def all_dirs(self) -> tuple[Path, ...]:
        return (self.data_dir, self.uploads_dir, self.artifacts_dir, self.logs_dir, self.cache_dir)

    def ensure_dirs(self) -> None:
        """创建数据目录。失败要明确报错——静默退到临时目录会让用户以为数据存住了。"""
        for path in self.all_dirs:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigError(f"无法创建数据目录 {path}：{exc}") from exc
            if not os.access(path, os.W_OK):
                raise ConfigError(f"数据目录不可写：{path}")

    def as_flask_config(self) -> dict:
        uri = f"sqlite:///{self.db_path}"
        return {
            "SQLALCHEMY_DATABASE_URI": uri,
            "SQLALCHEMY_ENGINE_OPTIONS": {
                # check_same_thread=False：Web 请求线程与后台 worker 线程会共用连接池，
                # 而 pysqlite 默认禁止跨线程使用连接。SQLAlchemy 的连接池保证了
                # 同一时刻只有一个线程持有某个连接，所以关掉这个检查是安全的。
                "connect_args": {"timeout": 15, "check_same_thread": False},
                # 长连接在 SQLite 上可能被外部进程（备份、迁移）打断，
                # 取用前 ping 一次可以避免拿到失效连接
                "pool_pre_ping": True,
            },
            "KB_DATA_DIR": str(self.data_dir),
            "KB_DB_PATH": str(self.db_path),
            "KB_UPLOADS_DIR": str(self.uploads_dir),
            "KB_ARTIFACTS_DIR": str(self.artifacts_dir),
            "KB_LOGS_DIR": str(self.logs_dir),
            "KB_CACHE_DIR": str(self.cache_dir),
            "SECRET_KEY": self.secret_key,
            "KB_AUTH_MODE": self.auth_mode,
            "KB_MAX_UPLOAD_MB": self.max_upload_mb,
            "KB_WORKER_EMBEDDED": self.worker_embedded,
            "MAX_CONTENT_LENGTH": self.max_upload_mb * 1024 * 1024,
            "DEBUG": self.debug,
        }


def _resolve_secret_key(data_dir: Path) -> tuple[bytes, str]:
    """密钥解析顺序：环境变量 > 磁盘上的 secret.key > 新生成。

    这里用 Fernet 密钥格式（44 字符 urlsafe base64，32 字节）。它同时被用作
    Flask 的 SECRET_KEY 和加密模型 API Key 的 Fernet key——少一个要保管的秘密。
    """
    from cryptography.fernet import Fernet

    env_key = _env("KB_SECRET_KEY")
    if env_key:
        try:
            Fernet(env_key.encode())
        except Exception as exc:
            raise ConfigError(
                "KB_SECRET_KEY 不是合法的 Fernet 密钥。生成一个：\n"
                "  python -c \"from cryptography.fernet import Fernet;"
                ' print(Fernet.generate_key().decode())"'
            ) from exc
        return env_key.encode(), "env:KB_SECRET_KEY"

    key_path = data_dir / "secret.key"
    if key_path.is_file():
        content = key_path.read_bytes().strip()
        if content:
            try:
                Fernet(content)
            except Exception as exc:
                raise ConfigError(
                    f"{key_path} 存在但不是合法的 Fernet 密钥。\n"
                    "如果这是误创建的，删掉它重启即可重新生成——"
                    "但注意：删除会导致已加密存库的 API Key 无法解密，需要重新填写。"
                ) from exc
            return content, f"file:{key_path}"

    key = Fernet.generate_key()
    try:
        # O_CREAT|O_EXCL + 0600：避免竞态，也避免密钥短暂地以宽权限存在
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
            fh.write(b"\n")
    except FileExistsError:
        # 另一个进程抢先创建了，用它的
        return key_path.read_bytes().strip(), f"file:{key_path}"
    except OSError as exc:
        raise ConfigError(
            f"无法写入密钥文件 {key_path}：{exc}\n"
            "可以改为通过 KB_SECRET_KEY 环境变量提供。"
        ) from exc
    return key, f"generated:{key_path}"


def load_boot_config(config_file: str | os.PathLike[str] | None = None) -> BootConfig:
    """解析启动级配置。优先级：环境变量 > config.toml > 代码默认值。"""
    load_dotenv(override=False)

    data_dir = _expand(_env("KB_DATA_DIR", DEFAULT_DATA_DIR))

    # config.toml 默认在数据目录下，但定位它本身要先知道数据目录，
    # 所以这里有一个先有鸡还是先有蛋的小循环——用两段式解决：
    # 先按环境变量定位一次配置文件，读到的值再覆盖。
    toml_path = _expand(config_file) if config_file else data_dir / "config.toml"
    toml = _load_toml(toml_path)

    if "data_dir" in toml:
        data_dir = _expand(toml["data_dir"])
        # 数据目录变了，配置文件也跟着走
        if not config_file:
            toml_path = data_dir / "config.toml"
            toml = _load_toml(toml_path)

    db_path = _expand(_env("KB_DB_PATH") or toml.get("db_path") or data_dir / DEFAULT_DB_NAME)

    host = _env("KB_HOST") or str(toml.get("host", "127.0.0.1"))
    port = _env_int("KB_PORT", int(toml.get("port", 5000)))
    debug = _env_bool("KB_DEBUG", bool(toml.get("debug", False)))
    log_level = (_env("KB_LOG_LEVEL") or str(toml.get("log_level", "INFO"))).upper()
    worker_embedded = _env_bool(
        "KB_WORKER_EMBEDDED", bool(toml.get("worker_embedded", True))
    )
    auth_mode = (_env("KB_AUTH_MODE") or str(toml.get("auth_mode", "apikey"))).lower()
    max_upload_mb = _env_int("KB_MAX_UPLOAD_MB", int(toml.get("max_upload_mb", 100)))

    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"KB_LOG_LEVEL 取值非法：{log_level!r}")
    if auth_mode not in {"apikey", "none"}:
        raise ConfigError(
            f"KB_AUTH_MODE 取值非法：{auth_mode!r}（可选 apikey / none）。"
            "none 只在完全可信的本机环境使用。"
        )

    cfg = BootConfig(
        data_dir=data_dir,
        db_path=db_path,
        host=host,
        port=port,
        debug=debug,
        log_level=log_level,
        worker_embedded=worker_embedded,
        auth_mode=auth_mode,
        max_upload_mb=max_upload_mb,
        secret_key=b"",  # 下面填
        secret_key_source="",
        seed_papers_root=str(toml.get("papers_root", DEFAULT_PAPERS_ROOT)),
        seed_notes_root=str(toml.get("notes_root", DEFAULT_NOTES_ROOT)),
        seed_codes_root=str(toml.get("codes_root", DEFAULT_CODES_ROOT)),
    )

    cfg.ensure_dirs()
    key, source = _resolve_secret_key(data_dir)

    final = BootConfig(
        data_dir=cfg.data_dir,
        db_path=cfg.db_path,
        host=cfg.host,
        port=cfg.port,
        debug=cfg.debug,
        log_level=cfg.log_level,
        worker_embedded=cfg.worker_embedded,
        auth_mode=cfg.auth_mode,
        max_upload_mb=cfg.max_upload_mb,
        secret_key=key,
        secret_key_source=source,
        seed_papers_root=cfg.seed_papers_root,
        seed_notes_root=cfg.seed_notes_root,
        seed_codes_root=cfg.seed_codes_root,
    )
    final.ensure_dirs()
    return final


def generate_secret_key() -> str:
    """供 CLI / 文档使用。"""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def random_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)
