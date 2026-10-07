#!/usr/bin/env bash
# 把一批论文走完整个流程：索引 -> 向量化 -> 关联代码 -> 生成笔记。
#
# 各阶段都是幂等的（已完成的会跳过），所以中途中断后直接重跑即可，
# 不会重复下载、重复索引或重复计费。
#
# 用法：
#   bash scripts/run_pipeline.sh              # 全部阶段
#   bash scripts/run_pipeline.sh index embed  # 只跑指定阶段

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

KB="pipenv run flask --app wsgi kb"
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=(index embed code read)
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 等后台的导入任务结束。判断依据是进程还在不在，而不是日志内容——
# 日志有缓冲，看日志会误判成「已经停了」。
if pgrep -f "kb import --list" > /dev/null; then
    log "批量导入仍在进行，等待其结束…"
    while pgrep -f "kb import --list" > /dev/null; do
        sleep 30
    done
    log "导入已结束"
fi

run_stage() {
    local stage="$1"
    log "=========== 开始：$stage ==========="
    case "$stage" in
        index)
            $KB index --wait 2>&1 | tail -5
            ;;
        embed)
            # 用 python 直接调，因为 embed 目前只有任务入口没有 CLI 包装
            pipenv run python - <<'PY' 2>&1 | tail -5
import warnings; warnings.filterwarnings("ignore")
from kb import create_app
app = create_app()
with app.app_context():
    from kb.services.embedding import embed_pending, embedding_stats
    print("嵌入:", embed_pending())
    print("状态:", embedding_stats())
PY
            ;;
        code)
            $KB code --all 2>&1 | tail -20
            ;;
        read)
            $KB read --all 2>&1 | tail -20
            ;;
        *)
            log "未知阶段：$stage"
            ;;
    esac
    log "=========== 结束：$stage ==========="
}

for stage in "${STAGES[@]}"; do
    run_stage "$stage"
done

log "全部阶段完成"
