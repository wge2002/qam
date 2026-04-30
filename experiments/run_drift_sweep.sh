#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

RUN_GROUP="${RUN_GROUP:-drift-diagnostics-r5-$(date +%m%d-%H%M)}"
ENV_NAME="${ENV_NAME:-cube-double-play-singletask-task2-v0}"
SEED="${SEED:-10001}"
WANDB_ENTITY="${WANDB_ENTITY:-xxxyyymmm}"

COMMON_ARGS=(
  --agent=agents/drift.py
  --seed="${SEED}"
  --sparse=False
  --horizon_length=5
  --agent.action_chunking=True
  --agent.behavior_support_k=0
  --agent.actor_hidden_dims="(512, 512, 512, 512)"
  --agent.value_hidden_dims="(512, 512, 512, 512)"
  --agent.batch_size=256
  --agent.num_qs=10
  --offline_steps=1000000
  --online_steps=0
  --log_interval=5000
  --eval_interval=50000
  --save_interval=50000
  --eval_episodes=50
)

run_one() {
  local tag="$1"
  local env_name="$2"
  shift 2

  local full_tag="${tag}"
  if [[ "${env_name}" == *"task1"* ]]; then
    full_tag="${tag}_TASK1"
  elif [[ "${env_name}" == *"task2"* ]]; then
    full_tag="${tag}_TASK2"
  fi

  echo "============================================================"
  echo "Starting ${full_tag}"
  echo "Run group: ${RUN_GROUP}"
  echo "Env: ${env_name}"
  echo "Seed: ${SEED}"
  echo "Extra args: $*"
  echo "============================================================"

  WANDB_ENTITY="${WANDB_ENTITY}" python main.py \
    --run_group="${RUN_GROUP}" \
    --tags="${full_tag}" \
    --env_name="${env_name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# Round 5: diagnostic reruns with extra vector-direction logging.
# These are not a broad sweep. They compare:
# 1) task1 successful NOPESS regime,
# 2) task2 with the same NOPESS regime,
# 3) task2 current best region.

run_one DRIFT_R5_NOPESS_B02_L02_T075 \
  cube-double-play-singletask-task1-v0 \
  --agent.drift_tau=0.75 \
  --agent.drift_beta=0.2 \
  --agent.drift_lambda_pi=0.2 \
  --agent.actor_pessimism_coef=0.0 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.05

run_one DRIFT_R5_NOPESS_B02_L02_T075 \
  cube-double-play-singletask-task2-v0 \
  --agent.drift_tau=0.75 \
  --agent.drift_beta=0.2 \
  --agent.drift_lambda_pi=0.2 \
  --agent.actor_pessimism_coef=0.0 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.05

run_one DRIFT_R5_CTRL_PESS05_B10_L10_STEP025 \
  cube-double-play-singletask-task2-v0 \
  --agent.drift_tau=0.50 \
  --agent.drift_beta=1.0 \
  --agent.drift_lambda_pi=1.0 \
  --agent.actor_pessimism_coef=0.5 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.025
