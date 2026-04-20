# DRIFT 实验小结
任务统一为 `cube-double-play-singletask-task1-v0`、`seed=10001`、`horizon_length=5`、`agent.action_chunking=True`、`online_steps=0`。

## 实验 1：原始 DRIFT baseline
命令核心：
```bash
MUJOCO_GL=egl python main.py \
  --run_group=reproduce \
  --agent=agents/drift.py \
  --tags=DRIFT \
  --seed=10001 \
  --env_name=cube-double-play-singletask-task1-v0 \
  --sparse=False \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --online_steps=0
```
评估结果：
| step | success | return | length |
| ---: | ---: | ---: | ---: |
| 50k | 0.76 | -278.20 | 246.70 |
| 100k | 0.48 | -483.52 | 370.92 |
| 200k | 0.36 | -496.04 | 389.20 |
| 300k | 0.26 | -637.26 | 424.02 |
| 400k | 0.22 | -561.24 | 436.60 |
| 500k | 0.14 | -642.28 | 462.22 |

关键现象：
- 成功率从 50k 的 0.76 下降到 500k 的 0.14，后 5 次评估平均只有 0.184。
- `actor/support_mse` 从约 0.033 降到约 0.014，说明 actor 越来越接近数据动作，但任务成功率反而下降。
- `actor/behavior_score_rms` 大约 1.9 到 2.9，长期高于 `actor/q_score_rms` 的约 0.38 到 0.79。
- `actor/target_step_rms` 大约 0.10 到 0.15，没有出现超大更新。

诊断：
baseline 不是“学不动”，而是越训练越被行为分布/KDE 项拉向不一定能完成任务的动作区域。这里的证据是 support MSE 下降与 success 下降同时发生。critic 的值本身没有明显发散到无法解释的程度，所以不能只归因于 critic 崩了。

## 实验 2：激进 Q-gradient 放大
命令核心：
```bash
PYTHONNOUSERSITE=1 MUJOCO_GL=egl /home/fine/nvme0n1/conda-envs/qam/bin/python main.py \
  --run_group=reproduce_drift_tune \
  --agent=agents/drift.py \
  --tags=DRIFT_QNORM_BETA02_TAU025 \
  --seed=10001 \
  --env_name=cube-double-play-singletask-task1-v0 \
  --sparse=False \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --online_steps=0 \
  --eval_interval=10000 \
  --save_interval=10000 \
  --agent.normalize_q_grad=True \
  --agent.drift_beta=0.2 \
  --agent.drift_tau=0.25
```
观察结果：
- 10k、20k、30k、40k、50k、60k 的 success 都是 0.0。
- `actor/q_score_rms` 被放大到约 4。
- `actor/behavior_score_rms` 约 17，乘上 `beta=0.2` 后有效量级约 3.4。
- `actor/policy_score_rms` 约 18，默认 `lambda_pi=1` 时 policy repulsion 量级过大。
- `actor/target_step_rms` 约 0.85，远高于 baseline 的约 0.10。

诊断：
这条实验验证了“直接加强 critic action-gradient”不是稳妥方向。critic 的局部 action-gradient 可能含噪声、尺度不稳，或者在离线数据支撑外给出错误方向；`normalize_q_grad=True` 加上较低 `tau=0.25` 后，actor 更新步长明显过大，早期就把策略推坏。
因此，critic action-gradient 的质量不好是主要风险之一，但结论不是“critic 完全没用”，而是“不能让局部梯度主导大步 actor 更新”。如果 critic 还能排序动作价值，后续更适合用多候选动作的 value selection，而不是继续放大梯度。

## 实验 3：TOP3_REWEIGHT_B02_L02
命令核心：
```bash
/home/fine/nvme0n1/conda-envs/qam/bin/python main.py \
  --run_group=reproduce_drift_top3 \
  --agent=agents/drift.py \
  --tags=TOP3_REWEIGHT_B02_L02 \
  --seed=10001 \
  --env_name=cube-double-play-singletask-task1-v0 \
  --sparse=False \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --online_steps=0 \
  --eval_interval=50000 \
  --save_interval=0 \
  --agent.drift_beta=0.2 \
  --agent.drift_tau=0.75 \
  --agent.drift_lambda_pi=0.2
```

评估结果：
| step | success | return | length |
| ---: | ---: | ---: | ---: |
| 50k | 0.14 | -498.32 | 451.26 |
| 100k | 0.34 | -389.00 | 367.00 |
| 200k | 0.42 | -349.10 | 342.78 |
| 350k | 0.60 | -287.62 | 287.52 |
| 500k | 0.52 | -348.26 | 327.92 |
| 650k | 0.62 | -311.16 | 298.06 |
| 750k | 0.64 | -273.66 | 270.72 |
| 1000k | 0.42 | -365.88 | 348.32 |

关键现象：
- 最高 success 达到 0.64，出现在 750k。
- 500k 时 success 是 0.52，明显高于 baseline 500k 的 0.14。
- 后 5 次评估平均 success 约 0.448，也明显高于 baseline 后 5 次的 0.184。
- `actor/target_step_rms` 约 0.027 到 0.037，比 baseline 更小，也远小于激进 Q-gradient 实验的约 0.85。
- `actor/q_score_rms` 约 0.82 到 1.00，高于 baseline；`actor/behavior_score_rms` 约 4.3 到 5.3，但乘上 `beta=0.2` 后有效量级约 0.86 到 1.06。
- `actor/policy_score_rms` 约 0.53 到 1.21，乘上 `lambda_pi=0.2` 后有效量级约 0.11 到 0.24。

诊断：
这条实验是目前最有价值的正向证据。它没有放大 critic gradient，而是把行为项和 policy repulsion 的有效权重降下来，同时用较温和的 `tau=0.75` 控制更新尺度。结果上，它没有 baseline 那种一路坠落，500k 之后仍能多次达到 0.5 到 0.64 的 success。
这说明：baseline 的失败更可能来自 actor 目标中各个 score 的相对权重不合适，而不是 critic 完全不可用。降低 `drift_beta` 和 `drift_lambda_pi` 能让 actor 少被不必要的约束/排斥拖偏，同时避免激进 Q-gradient 那种大步离线外推。

## 实验 4：TOP3_BEST32（进行中）
命令核心：
```bash
/home/fine/nvme0n1/conda-envs/qam/bin/python main.py \
  --run_group=reproduce_drift_top3 \
  --agent=agents/drift.py \
  --tags=TOP3_BEST32 \
  --seed=10001 \
  --env_name=cube-double-play-singletask-task1-v0 \
  --sparse=False \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --online_steps=0 \
  --eval_interval=50000 \
  --save_interval=0 \
  --agent.drift_beta=0.2 \
  --agent.drift_tau=0.75 \
  --agent.drift_lambda_pi=0.2 \
  --agent.best_of_n=32
```

理论依据：
实验 2 说明 critic 的局部 action-gradient 不能被粗暴放大；实验 3 说明 `drift_beta=0.2`、`drift_lambda_pi=0.2`、`drift_tau=0.75` 这组训练配比更稳。实验 4 因此不再加强 actor 更新中的 action-gradient，而是在评估/采样时把 `best_of_n` 从默认 8 提高到 32，让 actor 生成更多候选动作，再由 critic value ranking 选动作。

这条实验检验的是：critic 的局部梯度可能不可靠，但 critic 的相对排序是否仍有用。如果 `TOP3_BEST32` 的 success 明显高于实验 3，说明 value ranking 能从更多 actor samples 中挑出更好的动作；如果提升不明显，说明瓶颈更可能在 actor 候选分布本身，而不是候选数量。
