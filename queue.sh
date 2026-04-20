#!/usr/bin/env bash
#
# 按顺序串行执行 qam/main.py 的多次训练。
#
# jobs 文件每一行就是追加到 `python main.py` 后面的完整参数,
# 因此可以在不同 job 间切换算法(--agent=agents/<file>.py)
# 以及任意超参(--agent.<key>=<value>)。
#
# 用法:
#   ./queue.sh <jobs 文件>      # 每行一个 job
#   ./queue.sh -                # 从 stdin 读取 jobs
#
# 空行和以 `#` 开头的注释行会被忽略。
#
# 环境变量:
#   PYTHON_BIN     python 解释器路径(默认:/home/fine/anaconda3/envs/dsrl/bin/python)
#   MUJOCO_GL      MuJoCo 渲染后端(默认:egl)
#   WANDB_ENTITY   可选, W&B entity/team/user; 留空则使用 wandb 默认配置
#   LOG_DIR        单个 job 日志目录(默认:qam/logs/queue_<时间戳>)
#   STOP_ON_FAIL   设为 "1" 则第一个失败的 job 之后终止整条队列(默认:0,继续跑)
#
# jobs.txt 示例:
#   --run_group=qam_v1 --agent=agents/qam.py   --tags=QAM  --seed=0 --env_name=cube-triple-play-singletask-task2-v0 --agent.inv_temp=3.0
#   --run_group=fql_v1 --agent=agents/fql.py   --tags=FQL  --seed=0 --env_name=cube-triple-play-singletask-task2-v0 --agent.alpha=300.0
#   --run_group=drift1 --agent=agents/drift.py --tags=DFT  --seed=0 --env_name=cube-triple-play-singletask-task2-v0 --agent.tau=0.75 --agent.lambda_pi=1.0
#
# 挂后台运行(终端关了也不会断):
#   nohup ./queue.sh jobs.txt > queue_stdout.log 2>&1 &
# 或者放到 tmux 里:
#   tmux new -s queue 'bash queue.sh jobs.txt'

set -u

QAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$QAM_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/fine/anaconda3/envs/dsrl/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  export WANDB_ENTITY
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$QAM_DIR/logs/queue_$STAMP}"
mkdir -p "$LOG_DIR"

STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <jobs 文件 | ->" >&2
  exit 2
fi

if [[ "$1" == "-" ]]; then
  JOBS_SRC=/dev/stdin
else
  JOBS_SRC="$1"
  if [[ ! -f "$JOBS_SRC" ]]; then
    echo "找不到 jobs 文件: $JOBS_SRC" >&2
    exit 2
  fi
fi

SUMMARY="$LOG_DIR/summary.log"
JOBS_SNAPSHOT="$LOG_DIR/jobs.snapshot.txt"
if [[ "$JOBS_SRC" == "/dev/stdin" ]]; then
  cat "$JOBS_SRC" > "$JOBS_SNAPSHOT"
else
  cp "$JOBS_SRC" "$JOBS_SNAPSHOT"
fi

echo "队列目录  : $QAM_DIR"
echo "日志目录  : $LOG_DIR"
echo "汇总日志  : $SUMMARY"
echo "jobs快照 : $JOBS_SNAPSHOT"
echo "python    : $PYTHON_BIN"
echo "wandb实体 : ${WANDB_ENTITY:-<wandb默认>}"
echo "=== 队列开始于 $(date +%F_%T) ===" | tee -a "$SUMMARY"

CUR_PID=""
cleanup() {
  echo "[$(date +%F_%T)] 收到中断信号;杀掉当前 job pid=${CUR_PID:-无}" | tee -a "$SUMMARY"
  if [[ -n "$CUR_PID" ]]; then
    kill "$CUR_PID" 2>/dev/null || true
    wait "$CUR_PID" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup INT TERM

idx=0
fails=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line#"${line%%[![:space:]]*}"}"   # 去掉行首空白
  [[ -z "$line" || "$line" == \#* ]] && continue
  idx=$((idx + 1))
  name="job_$(printf '%03d' "$idx")"
  log_file="$LOG_DIR/${name}.log"
  echo "[$(date +%F_%T)] 开始 $name -> $log_file" | tee -a "$SUMMARY"
  echo "  命令: $PYTHON_BIN main.py $line" | tee -a "$SUMMARY"

  # 这里的词分割是有意的,让每个 flag 成为独立的 argv
  # shellcheck disable=SC2086
  $PYTHON_BIN main.py $line > "$log_file" 2>&1 &
  CUR_PID=$!
  wait "$CUR_PID"
  rc=$?
  CUR_PID=""

  echo "[$(date +%F_%T)] 结束 $name (退出码=$rc)" | tee -a "$SUMMARY"
  if [[ "$rc" -ne 0 ]]; then
    fails=$((fails + 1))
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      echo "STOP_ON_FAIL=1,中止后续 job" | tee -a "$SUMMARY"
      exit "$rc"
    fi
  fi
done < "$JOBS_SNAPSHOT"

echo "=== 队列结束于 $(date +%F_%T);共执行 $idx 个 job,失败 $fails 个 ===" | tee -a "$SUMMARY"
exit $(( fails > 0 ? 1 : 0 ))
