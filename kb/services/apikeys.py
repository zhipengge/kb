"""对外接口的 API Key 管理。

Key 的形态是 ``kb_<前缀>_<随机串>``：

  * ``kb_`` 前缀让它在日志、代码、配置文件里一眼可辨——也方便做日志脱敏，
    只要匹配这个前缀就能把它打码。
  * 中间 8 位前缀是**明文存储**的，用于列表展示和吊销。
    「我要吊销哪一把」这个问题，必须能在不接触哈希的情况下回答。
  * 剩余部分是随机串，只存 sha256。数据库泄漏也拿不到可用的 Key。

哈希用 sha256 而不是 bcrypt/argon2：API Key 本身就是 32 字节的高熵随机串，
不存在被字典攻击的前提，用慢哈希只会让每个请求白白多花几十毫秒。
（用户密码是另一回事——低熵、会被撞库，那里必须用慢哈希。）
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import timedelta

from ..extensions import db
from ..models import ApiKey
from ..models.base import utcnow

log = logging.getLogger(__name__)

_PREFIX = "kb"
_PREFIX_LEN = 8
_SECRET_BYTES = 32


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_key(
    name: str,
    scopes: list[str] | None = None,
    *,
    expires_in_days: int | None = None,
    rate_limit: str | None = None,
) -> tuple[str, ApiKey]:
    """签发一个新 Key。返回 ``(明文, 记录)``——明文只在这一刻存在。"""
    identifier = secrets.token_hex(_PREFIX_LEN // 2)  # 8 个十六进制字符
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    plaintext = f"{_PREFIX}_{identifier}_{secret}"

    row = ApiKey(
        name=name,
        prefix=identifier,
        key_hash=_hash(plaintext),
        scopes=list(scopes or ["read"]),
        rate_limit=rate_limit,
        expires_at=utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
    )
    db.session.add(row)
    db.session.commit()

    log.info("已签发 API Key %s（权限：%s）", row.prefix, ",".join(row.scopes or []))
    return plaintext, row


def verify_key(plaintext: str) -> ApiKey | None:
    """校验 Key。成功返回记录，失败返回 None。

    用 ``compare_digest`` 做常量时间比较——虽然哈希值相等与否本身
    不构成有意义的旁路（攻击者无法逐字节试探哈希），但保持一致的做法
    成本为零，也免去日后审查时解释「这里为什么可以不用」。
    """
    if not plaintext or not plaintext.startswith(f"{_PREFIX}_"):
        return None

    candidate = _hash(plaintext)
    row = db.session.query(ApiKey).filter(ApiKey.key_hash == candidate).one_or_none()

    # 也允许按前缀查一次来做常量时间兜底（前缀是明文，可能的候选很少）
    if row is None:
        parts = plaintext.split("_", 2)
        if len(parts) >= 2:
            fallback = db.session.query(ApiKey).filter(ApiKey.prefix == parts[1]).one_or_none()
            if fallback is not None and hmac.compare_digest(fallback.key_hash, candidate):
                row = fallback
    if row is None or not row.is_active:
        return None
    return row


def touch_key(row: ApiKey) -> None:
    """记录一次使用。"""
    row.last_used_at = utcnow()
    row.call_count = (row.call_count or 0) + 1
    db.session.commit()


def revoke_key(row: ApiKey) -> None:
    row.revoked_at = utcnow()
    db.session.commit()


__all__ = ["create_key", "revoke_key", "touch_key", "verify_key"]
