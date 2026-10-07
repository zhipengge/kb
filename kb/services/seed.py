"""首次启动时的数据播种。

只做两件事：把启动级配置里的路径写进设置表（让用户第一次打开设置页时
看到的就是自己填的值），以及建一组「阅读状态」标签。

**刻意不预置方法/主题类标签**：预置一份别人想象中的分类体系，
用户第一件事就是删它。词表应该从用户自己的文献里长出来，
而不是从一开始就被塞满。阅读状态是唯一例外——它描述的是「我和这篇论文的关系」，
不是论文本身，跟领域无关，且没有它就没法做最基本的「待读清单」。
"""

from __future__ import annotations

import logging

from flask import current_app

from ..extensions import db
from ..models import Setting, Tag
from ..models.tag import DIM_STATUS
from ..settings import Settings

log = logging.getLogger(__name__)

# 阅读状态：固定、通用、与领域无关
_STARTER_STATUS_TAGS = [
    ("待读", "to-read", "#94a3b8", "还没开始读"),
    ("在读", "reading", "#3b82f6", "正在读"),
    ("已读", "read", "#22c55e", "读完了"),
    ("待复现", "to-reproduce", "#f59e0b", "打算跑一遍代码"),
    ("已复现", "reproduced", "#8b5cf6", "已经复现过"),
    ("重点", "starred", "#ef4444", "需要反复回看的"),
]


def _seed_paths(settings: Settings) -> bool:
    """把启动级配置里的路径写进设置表。

    只在「用户从未改过」时写入——否则用户改了路径之后，
    改 .env 或 config.toml 又会被无声地覆盖回去。
    """
    cfg = current_app.extensions["kb_boot_config"]
    changed = False

    seeds = {
        "paths.papers_roots": [cfg.seed_papers_root],
        "paths.notes_root": cfg.seed_notes_root,
        "paths.codes_root": cfg.seed_codes_root,
    }
    existing = {row.key for row in db.session.query(Setting).all()}

    for key, value in seeds.items():
        if key not in existing:
            db.session.add(Setting(key=key, value=value, updated_by="bootstrap"))
            changed = True

    if changed:
        db.session.commit()
        settings.invalidate()
        log.info("已写入默认路径配置")

    return changed


def _seed_status_tags() -> int:
    created = 0
    for name, slug, color, description in _STARTER_STATUS_TAGS:
        exists = (
            db.session.query(Tag)
            .filter_by(dimension=DIM_STATUS, slug=slug)
            .one_or_none()
        )
        if exists is None:
            db.session.add(
                Tag(
                    name=name,
                    slug=slug,
                    dimension=DIM_STATUS,
                    color=color,
                    description=description,
                    is_auto=False,
                )
            )
            created += 1
    if created:
        db.session.commit()
        log.info("已创建 %d 个阅读状态标签", created)
    return created


def ensure_seed_data() -> None:
    """幂等：每次启动都调用，只在缺失时补齐。"""
    settings: Settings = current_app.extensions["kb_settings"]
    _seed_paths(settings)
    _seed_status_tags()


__all__ = ["ensure_seed_data"]
