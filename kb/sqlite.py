"""SQLite 接入层：PRAGMA 调优、启动自检、向量扩展加载。

这个模块承载了两个在真实环境里踩过的坑，值得写清楚：

**坑一：`PRAGMA journal_mode=WAL` 在事务内会静默失败。**
SQLite 不允许在事务中切换日志模式，此时它不报错，只是把当前模式原样返回
（通常是 `delete`），于是你以为开了 WAL，其实没有。Python 的 sqlite3 在执行
INSERT/UPDATE 后会隐式开启事务，所以「随手连上就跑 PRAGMA」很容易中招。
本模块的做法是在 SQLAlchemy 的 ``connect`` 事件里、连接刚建立还干净的时候
设置 PRAGMA——那时一定不在事务中。

**坑二：SQLite 的外键约束默认是关闭的。**
不显式 ``PRAGMA foreign_keys=ON`` 的话，所有 FK 声明都只是注释。必须每连接开启。

关于 9p 挂载（WSL 下的 /mnt、D:\\ 等）：实测 WAL 在这类挂载上可以正常工作，
所以数据目录放在 Windows 盘上并不会坏，只是 9p 的 IO 明显慢。因此自检仍然保留
——它便宜，且能覆盖别的 WSL 版本/发行版上可能出现的差异——但结论只影响提示，
不影响能否启动。
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# PRAGMA
# --------------------------------------------------------------------------

def apply_pragmas(dbapi_conn, journal_mode: str = "WAL") -> dict:
    """在一个干净的 DBAPI 连接上设置 PRAGMA。返回实际生效的值。

    必须在事务外调用——见模块文档「坑一」。
    """
    cur = dbapi_conn.cursor()
    applied: dict[str, object] = {}
    try:
        # 注意：PRAGMA 分两类。`PRAGMA x=值` 多数**不返回结果行**，
        # 只有 `PRAGMA x` 才返回当前值。对设置型语句调用 fetchone() 会拿到 None。
        # 所以统一「先设置，再单独读回来」。
        def set_then_read(statement: str, readback: str):
            cur.execute(statement)
            row = cur.execute(readback).fetchone()
            return row[0] if row else None

        # 外键：SQLite 默认关闭，必须每连接开（坑二）
        applied["foreign_keys"] = bool(
            set_then_read("PRAGMA foreign_keys=ON", "PRAGMA foreign_keys")
        )

        # 忙等而非立刻报 "database is locked"。后台任务与 Web 请求会并发写，
        # 没有这个设置，瞬时锁冲突会直接变成 500。
        applied["busy_timeout"] = set_then_read(
            "PRAGMA busy_timeout=10000", "PRAGMA busy_timeout"
        )

        # 日志模式。WAL 允许「一写多读」并发。
        try:
            mode = cur.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            log.warning("设置 journal_mode=%s 失败（%s），回退到 DELETE", journal_mode, exc)
            mode = cur.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        applied["journal_mode"] = mode

        # synchronous=NORMAL 在 WAL 下是安全的（崩溃不会损坏数据库，
        # 最多丢失最近若干已提交事务），换来明显的写入吞吐提升。
        sync_value = 1 if mode == "wal" else 2  # NORMAL : FULL
        applied["synchronous"] = set_then_read(
            f"PRAGMA synchronous={sync_value}", "PRAGMA synchronous"
        )

        # 内存临时表 + 负数 cache_size = 以 KiB 为单位（此处约 32 MiB）
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("PRAGMA cache_size=-32000")
        cur.execute("PRAGMA mmap_size=268435456")  # 256 MiB

        # LIKE 的大小写敏感性：保持默认的「不敏感」，与用户直觉一致
        cur.execute("PRAGMA case_sensitive_like=OFF")
    finally:
        cur.close()
    return applied


def register_engine_pragmas(engine: Engine, journal_mode: str = "WAL") -> None:
    """把 PRAGMA 挂到 SQLAlchemy 引擎的 connect 事件上。"""

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):
        try:
            apply_pragmas(dbapi_conn, journal_mode=journal_mode)
        except Exception:
            # 连接级设置失败不应让整个应用起不来，但必须留下痕迹
            log.exception("设置 SQLite PRAGMA 时出错，数据库将以默认参数运行")


# --------------------------------------------------------------------------
# sqlite-vec
# --------------------------------------------------------------------------

def load_sqlite_vec(dbapi_conn) -> str | None:
    """尝试加载 sqlite-vec 扩展，返回版本号；失败返回 None。

    ``enable_load_extension`` 是 CPython 构建期选项，某些发行版/平台会关闭。
    关闭时不是崩溃，而是降级为 numpy 暴力检索——万级分块的暴力检索在
    单机上完全可以接受，所以这是可接受的退路。
    """
    if not hasattr(dbapi_conn, "enable_load_extension"):
        return None
    try:
        import sqlite_vec  # 延迟导入：没装也不该影响启动
    except ImportError:
        log.info("未安装 sqlite-vec，向量检索将使用 numpy 暴力检索")
        return None

    try:
        dbapi_conn.enable_load_extension(True)
        sqlite_vec.load(dbapi_conn)
        dbapi_conn.enable_load_extension(False)
        version = dbapi_conn.execute("SELECT vec_version()").fetchone()[0]
        return version
    except Exception:
        log.exception("加载 sqlite-vec 失败，向量检索将使用 numpy 暴力检索")
        with contextlib.suppress(Exception):
            # 失败路径上关闭扩展加载，失败本身无所谓——真正的问题已经在上面记了
            dbapi_conn.enable_load_extension(False)
        return None


def register_vec_loader(engine: Engine) -> dict:
    """把 sqlite-vec 加载挂到引擎上。返回可变的状态字典，供自检读取。"""
    state: dict = {"version": None, "enabled": False}

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):
        version = load_sqlite_vec(dbapi_conn)
        if version:
            state["version"] = version
            state["enabled"] = True

    return state


# --------------------------------------------------------------------------
# 启动自检
# --------------------------------------------------------------------------


@dataclass
class PreflightReport:
    """数据库能力自检结果。会原样显示在设置页与 /api/v1/system/health。"""

    db_path: str
    sqlite_version: str = ""
    journal_mode: str = ""
    foreign_keys: bool = False
    fts5: bool = False
    fts5_trigram: bool = False
    vector_enabled: bool = False
    vector_version: str | None = None
    writable: bool = False
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """能不能跑起来。注意：向量不可用不算致命——会降级到暴力检索。"""
        return self.writable and self.fts5 and not self._fatal_problems

    @property
    def _fatal_problems(self) -> list[str]:
        return [p for p in self.problems if p.startswith("[fatal]")]

    def to_dict(self) -> dict:
        return {
            "db_path": self.db_path,
            "sqlite_version": self.sqlite_version,
            "journal_mode": self.journal_mode,
            "foreign_keys": self.foreign_keys,
            "fts5": self.fts5,
            "fts5_trigram": self.fts5_trigram,
            "vector_enabled": self.vector_enabled,
            "vector_version": self.vector_version,
            "writable": self.writable,
            "ok": self.ok,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def _retry_on_lock(operation, *, attempts: int = 4, base_delay: float = 0.8):
    """执行一个可能因锁失败的操作，遇到锁就退避重试。

    返回 ``(成功, 错误信息, 错误是否是锁冲突)``。

    **区分「锁冲突」和「真的不行」非常重要。** 自检在应用启动时运行，
    而那时后台任务可能正在写库。之前直接把锁冲突报成
    「数据库不可写」甚或「此 SQLite 未启用 FTS5」——后者是完全错误的
    诊断，会让人跑去重新编译 SQLite，而真正的原因只是那一瞬间的并发。
    """
    import time

    last_error = ""
    for attempt in range(attempts):
        try:
            operation()
            return True, "", False
        except sqlite3.OperationalError as exc:
            last_error = str(exc)
            if "locked" not in last_error.lower() and "busy" not in last_error.lower():
                return False, last_error, False
            if attempt < attempts - 1:
                time.sleep(base_delay * (attempt + 1))
        except sqlite3.Error as exc:
            return False, str(exc), False
    return False, last_error, True


def preflight(db_path: str | Path, journal_mode: str = "WAL") -> PreflightReport:
    """打开数据库并逐项探测能力。

    这里**故意**用原生 sqlite3 而不是 SQLAlchemy：自检要在 ORM 之前跑，
    而且需要直接控制事务边界（否则会踩到 journal_mode 的坑）。
    """
    db_path = Path(db_path)
    report = PreflightReport(db_path=str(db_path))

    if not db_path.parent.exists():
        report.problems.append(f"[fatal] 数据库所在目录不存在：{db_path.parent}")
        return report

    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)  # autocommit
    except sqlite3.Error as exc:
        report.problems.append(f"[fatal] 无法打开数据库 {db_path}：{exc}")
        return report

    try:
        report.sqlite_version = sqlite3.sqlite_version

        # --- PRAGMA 必须最先做 ---
        #
        # 尤其是 busy_timeout：它必须在**任何可能碰到锁的操作之前**设好。
        # 之前的顺序是先测可写性、再设 PRAGMA，而可写性测试是在
        # busy_timeout=0 下跑的——只要那一刻有别的进程在写（比如后台任务
        # 正在跑），就会立刻失败并报出 `[fatal] 数据库不可写`。
        # 数据库其实完全正常，这个假警报会让人以为数据坏了。
        try:
            pragmas = apply_pragmas(conn, journal_mode=journal_mode)
            report.journal_mode = str(pragmas["journal_mode"])
            report.foreign_keys = bool(pragmas["foreign_keys"])
            if report.journal_mode != journal_mode.lower():
                report.problems.append(
                    f"journal_mode 请求 {journal_mode}，实际为 {report.journal_mode}"
                )
                report.notes.append(
                    "已降级为非 WAL 日志模式。功能不受影响，但并发写入性能下降，"
                    "且「一写多读」不再成立。常见于网络文件系统或只读挂载。"
                )
        except sqlite3.Error as exc:
            report.problems.append(f"[fatal] 设置 PRAGMA 失败：{exc}")

        # --- 可写性（此时 busy_timeout 已生效，短暂并发会重试而不是误报）---
        def _probe_write() -> None:
            # 拿一次写锁再放开，**不建表**。
            #
            # 这里以前是「建一张 _kb_preflight 再删掉」——那是真正的 DDL，
            # 会让**所有其它连接**的 schema 缓存失效。实测后果：批处理正在
            # 跑的时候，只要另开一个进程跑任何 CLI 命令（每次启动都跑自检），
            # 批处理里就会冒出 `no such table: chunks` 这种莫名其妙的报错，
            # 而且每次都发生在不同的论文上，看起来像是数据损坏。
            #
            # BEGIN IMMEDIATE 直接申请 RESERVED 锁，这才是「能不能写」本身；
            # 建表探测测的其实是「能不能改 schema」，两回事。
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")

        ok, error, locked = _retry_on_lock(_probe_write)
        if ok:
            report.writable = True
        elif locked:
            # 重试这么多次仍然拿不到锁，多半是另一个进程正长时间占着写锁
            report.problems.append(
                f"[fatal] 数据库被占用，暂时无法写入：{error}"
            )
            report.notes.append(
                "如果有批量任务正在运行，等它结束后重启即可。"
                "若持续出现，检查是否有进程卡住不释放锁。"
            )
        else:
            report.problems.append(f"[fatal] 数据库不可写：{error}")

        # --- FTS5 / trigram ---
        #
        # 在**内存库**里试建，不碰真库。两个理由：
        #   * 又一次避免 DDL 使别人的 schema 失效（同上）；
        #   * 探测的是「这个 SQLite 构建带不带 fts5 模块」，与具体这个库文件无关，
        #     所以内存库得到的结论完全等价。顺带也不再可能出现锁冲突，
        #     那种「无法验证 FTS5（数据库被占用）」的中间状态就此消失。
        def _probe_memory(ddl: str, query: str, needle: str) -> int:
            probe = sqlite3.connect(":memory:")
            try:
                probe.execute(ddl)
                probe.execute(f"INSERT INTO t(x) VALUES ({needle})")
                return probe.execute(query).fetchone()[0]
            finally:
                probe.close()

        try:
            _probe_memory(
                "CREATE VIRTUAL TABLE t USING fts5(x)",
                "SELECT count(*) FROM t",  # 建得出来就算支持
                "'probe'",
            )
            report.fts5 = True
        except sqlite3.Error as exc:
            report.problems.append(
                f"[fatal] 此 SQLite 未启用 FTS5（{exc}）。全文检索无法工作，"
                "需要重新编译 SQLite 或更换 Python 构建。"
            )

        if report.fts5:
            try:
                if _probe_memory(
                    "CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')",
                    "SELECT count(*) FROM t WHERE t MATCH '注意力'",
                    "'全连接层的注意力机制'",
                ):
                    report.fts5_trigram = True
                else:
                    report.problems.append("FTS5 trigram 分词器存在但检索无结果")
            except sqlite3.Error as exc:
                report.problems.append(
                    f"FTS5 trigram 分词器不可用（{exc}）。"
                    "中文检索将退化为 LIKE 扫描，短词检索质量下降。"
                )
                report.notes.append("trigram 需要 SQLite >= 3.34。")

        # --- sqlite-vec ---
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            report.vector_version = conn.execute("SELECT vec_version()").fetchone()[0]
            report.vector_enabled = True
        except Exception as exc:
            report.problems.append(f"sqlite-vec 不可用：{exc}")
            report.notes.append(
                "向量检索将降级为 numpy 暴力检索——单机万级分块下速度可接受，功能不受影响。"
            )
            with contextlib.suppress(Exception):
                conn.enable_load_extension(False)
    finally:
        conn.close()

    return report


def vacuum_into(db_path: str | Path, target: str | Path) -> None:
    """压缩备份数据库（不依赖 sqlite3 CLI，环境里通常没有）。"""
    src = sqlite3.connect(str(db_path))
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


__all__ = [
    "PreflightReport",
    "apply_pragmas",
    "load_sqlite_vec",
    "preflight",
    "register_engine_pragmas",
    "register_vec_loader",
    "vacuum_into",
]
