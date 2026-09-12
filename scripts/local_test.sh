#!/usr/bin/env bash
# ==============================================================================
#  Agentic-RAG 本地评测：拷脚本进容器 -> 跑评测 -> 清测试数据
#
#  为什么需要这个脚本：scripts/ 在 .dockerignore 里（镜像内没有），每次跑评测
#  都要手工 docker cp；而 evaluate.py 用 __file__ 反推项目根，落点必须是
#  /app/scripts/ 而不是 /tmp（放 /tmp 会 import 不到 app.*）。
#  另外 Git Bash 会把容器路径改写成 Windows 路径，MSYS_NO_PATHCONV=1 漏一次就失败。
#
#  用法：
#    ./scripts/local_test.sh                 # 导入检查 + 检索/路由评测(20条) + 清数据
#    ./scripts/local_test.sh --full          # 额外跑 generation/e2e + LLM Judge（慢、花钱）
#    ./scripts/local_test.sh --sample 50     # 换采样条数
#    ./scripts/local_test.sh --no-clean      # 跑完不清数据（想留证据时）
#    ./scripts/local_test.sh --clean-only    # 只清数据
#    ./scripts/local_test.sh --smoke         # 只做导入检查（不花 LLM 钱，改完评测模块先跑这个）
#    ./scripts/local_test.sh --help
#
#  注意：测试集会往 SQLite 里写患者/会话/问答记录，默认跑完自动清掉。
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER="agentic-rag-api"
COMPOSE_SVC="api"

# Git Bash 会把 /app/... 改写成 C:\...，docker cp 与 exec 全部依赖原始路径
export MSYS_NO_PATHCONV=1

# 容器内的落点。evaluate.py 依赖 PROJECT_ROOT=/app，所以不能放 /tmp
CTR_SCRIPTS="/app/scripts"
CTR_DATA="/app/data"
CTR_TESTSET="${CTR_DATA}/testset.csv"

DEFAULT_TESTSET="scripts/strategy_comparison/sampled_100_qa_rewritten_labeled.csv"

# ---- 默认参数 ----------------------------------------------------------------
SAMPLES=20
STAGES="retrieval routing"
STRATEGIES=""
LLM_JUDGE=""
DO_SMOKE=1
DO_EVAL=1
DO_CLEAN=1
CLEAN_ONLY=0
TESTSET="$DEFAULT_TESTSET"
EXTRA_ARGS=()

# ---- 输出 --------------------------------------------------------------------
c_ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
c_warn() { printf '\033[33m%s\033[0m\n' "$*"; }
c_err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
step()   { printf '\n\033[36m==> %s\033[0m\n' "$*"; }

# 打印文件头部注释块：从第 2 行到第二个分隔线为止，去掉行首的 "# "
usage() {
  awk 'NR>1 && /^# =+$/ { if (++n == 2) exit; next }
       NR>1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
  exit 0
}

# ---- 解析参数 ----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)       STAGES="retrieval routing generation e2e"; LLM_JUDGE="--llm-judge" ;;
    --smoke)      DO_SMOKE=1; DO_EVAL=0; DO_CLEAN=0 ;;
    --clean-only) DO_SMOKE=0; DO_EVAL=0; CLEAN_ONLY=1 ;;
    --no-clean)   DO_CLEAN=0 ;;
    --sample)     SAMPLES="$2"; shift ;;
    --stages)     STAGES="$2"; shift ;;
    --strategies) STRATEGIES="$2"; shift ;;
    --testset)    TESTSET="$2"; shift ;;
    --llm-judge)  LLM_JUDGE="--llm-judge" ;;
    -h|--help)    usage ;;
    *)            EXTRA_ARGS+=("$1") ;;
  esac
  shift
done

# ---- 前置检查 ----------------------------------------------------------------
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  c_err "容器 $CONTAINER 未运行。先执行： docker compose up -d"
  exit 1
fi

# 健康检查看 body 不看状态码：RAG 未就绪时 /api/health 同样返回 200
HEALTH="$(docker compose exec -T "$COMPOSE_SVC" python -c \
  "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=5)).get('status','?'))" \
  2>/dev/null | tr -d '\r' | tail -1 || echo "unreachable")"
if [[ "$HEALTH" != "ok" ]]; then
  c_warn "RAG 状态为 '$HEALTH'（不是 ok）。首次启动要加载 BGE + TCMNER，等几分钟再试。"
  [[ "$DO_EVAL" == "1" ]] && { c_err "评测需要 RAG 就绪，中止。"; exit 1; }
fi
c_ok "容器运行中，RAG 状态: $HEALTH"

# ---- 拷脚本进去 --------------------------------------------------------------
step "拷贝脚本到容器 $CTR_SCRIPTS"
docker compose exec -T -u root "$COMPOSE_SVC" mkdir -p "$CTR_SCRIPTS" "$CTR_DATA"

for f in evaluate.py verify_evaluation_imports.py clear_test_data.py; do
  [[ -f "scripts/$f" ]] || { c_warn "跳过（本地不存在）: scripts/$f"; continue; }
  docker cp "scripts/$f" "${CONTAINER}:${CTR_SCRIPTS}/"
  printf '    %s\n' "$f"
done

if [[ "$DO_EVAL" == "1" ]]; then
  if [[ ! -f "$TESTSET" ]]; then
    c_err "测试集不存在: $TESTSET"
    cat <<'EOF'
    该文件被 .gitignore 的 scripts/**/sampled_*.csv 挡住，新克隆的仓库里没有。
    重新生成：
      cd scripts/strategy_comparison
      python sample_testset.py              # 采样 + LLM 改写查询
      python label_routing_ground_truth.py  # 标注路由 ground truth
EOF
    exit 1
  fi
  docker cp "$TESTSET" "${CONTAINER}:${CTR_TESTSET}"
  printf '    测试集 -> %s\n' "$CTR_TESTSET"
fi
c_ok "拷贝完成"

# ---- 1. 导入检查 -------------------------------------------------------------
if [[ "$DO_SMOKE" == "1" ]]; then
  step "1/3 评测模块导入检查（18 个文件，不调 LLM）"
  docker compose exec -T -u root -w /app -e PYTHONIOENCODING=utf-8 \
    "$COMPOSE_SVC" python "${CTR_SCRIPTS}/verify_evaluation_imports.py" 2>&1 | tail -22
fi

# ---- 2. 评测 -----------------------------------------------------------------
if [[ "$DO_EVAL" == "1" ]]; then
  step "2/3 评测（stages: $STAGES；sample: $SAMPLES）"
  EVAL_ARGS=(--csv "$CTR_TESTSET" --stages $STAGES --sample "$SAMPLES" --detailed)
  [[ -n "$STRATEGIES" ]] && EVAL_ARGS+=(--strategies $STRATEGIES)
  [[ -n "$LLM_JUDGE"  ]] && EVAL_ARGS+=($LLM_JUDGE)
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && EVAL_ARGS+=("${EXTRA_ARGS[@]}")

  # 不用 tail 收尾：那会把输出全部缓冲到进程结束，评测跑几分钟期间屏幕上什么都
  # 没有，看起来像卡死。改成 tee 实时流式，同时留一份完整日志备查。
  EVAL_LOG="scripts/.local_test_last.log"
  printf '    python evaluate.py %s\n' "${EVAL_ARGS[*]}"
  printf '    完整输出同时写入 %s\n\n' "$EVAL_LOG"

  set +e
  docker compose exec -T -u root -w /app -e PYTHONIOENCODING=utf-8 \
    "$COMPOSE_SVC" python "${CTR_SCRIPTS}/evaluate.py" "${EVAL_ARGS[@]}" 2>&1 \
    | tee "$EVAL_LOG"
  EVAL_RC="${PIPESTATUS[0]}"
  set -e

  if [[ "$EVAL_RC" != "0" ]]; then
    c_err "evaluate.py 退出码 $EVAL_RC（完整输出见 $EVAL_LOG）"
  else
    printf '\n    摘要: '
    grep -a ">>> SUMMARY" "$EVAL_LOG" | tail -1 | sed 's/.*>>> SUMMARY: //'
  fi
fi

# ---- 3. 清数据 ---------------------------------------------------------------
if [[ "$CLEAN_ONLY" == "1" || "$DO_CLEAN" == "1" ]]; then
  step "3/3 清空测试数据（只动 qa_history.db，不碰 Milvus / Neo4j）"
  CLEAN_FLAGS=(--yes)
  [[ "$CLEAN_ONLY" == "1" ]] && CLEAN_FLAGS=(--dry-run)
  docker compose exec -T -e PYTHONIOENCODING=utf-8 \
    "$COMPOSE_SVC" python "${CTR_SCRIPTS}/clear_test_data.py" "${CLEAN_FLAGS[@]}" 2>&1 | tail -16
fi

echo
c_ok "全部完成。"
