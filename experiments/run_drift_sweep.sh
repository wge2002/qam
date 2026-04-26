#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

RUN_GROUP="${RUN_GROUP:-drift-param8-$(date +%m%d-%H%M)}"
ENV_NAME="${ENV_NAME:-cube-double-play-singletask-task2-v0}"
SEED="${SEED:-10001}"
WANDB_ENTITY="${WANDB_ENTITY:-xxxyyymmm}"

COMMON_ARGS=(
  --agent=agents/drift.py
  --seed="${SEED}"
  --env_name="${ENV_NAME}"
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
  shift

  echo "============================================================"
  echo "Starting ${tag}"
  echo "Run group: ${RUN_GROUP}"
  echo "Env: ${ENV_NAME}"
  echo "Seed: ${SEED}"
  echo "Extra args: $*"
  echo "============================================================"

  WANDB_ENTITY="${WANDB_ENTITY}" python main.py \
    --run_group="${RUN_GROUP}" \
    --tags="${tag}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# No baseline in this script. Baseline was:
# tau=0.75 beta=1.0 lambda_pi=1.0 bandwidth=0.25 step=0.05.
# Keep behavior_support_k=0 for this round to avoid kNN support bias.

run_one DRIFT_TAU050_BETA100 \
  --agent.drift_tau=0.50 \
  --agent.drift_beta=1.0 \
  --agent.drift_lambda_pi=1.0 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.05

run_one DRIFT_TAU035_BETA100 \
  --agent.drift_tau=0.35 \
  --agent.drift_beta=1.0 \
  --agent.drift_lambda_pi=1.0 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.05

run_one DRIFT_TAU075_BETA050 \
  --agent.drift_tau=0.75 \
  --agent.drift_beta=0.5 \
  --agent.drift_lambda_pi=1.0 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.05

run_one DRIFT_TAU050_BETA050 \
  --agent.drift_tau=0.50 \
  --agent.drift_beta=0.5 \
  --agent.drift_lambda_pi=1.0 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.05

run_one DRIFT_TAU050_BETA050_BW010 \
  --agent.drift_tau=0.50 \
  --agent.drift_beta=0.5 \
  --agent.drift_lambda_pi=1.0 \
  --agent.kernel_bandwidth=0.10 \
  --agent.transport_step_size=0.05

run_one DRIFT_TAU050_BETA050_BW050 \
  --agent.drift_tau=0.50 \
  --agent.drift_beta=0.5 \
  --agent.drift_lambda_pi=1.0 \
  --agent.kernel_bandwidth=0.50 \
  --agent.transport_step_size=0.05

run_one DRIFT_TAU050_BETA050_STEP025 \
  --agent.drift_tau=0.50 \
  --agent.drift_beta=0.5 \
  --agent.drift_lambda_pi=1.0 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.025

run_one DRIFT_TAU050_BETA050_STEP100 \
  --agent.drift_tau=0.50 \
  --agent.drift_beta=0.5 \
  --agent.drift_lambda_pi=1.0 \
  --agent.kernel_bandwidth=0.25 \
  --agent.transport_step_size=0.10
