### install 
conda create -n qam python=3.10
conda activate qam
pip install -U "jax[cuda12]"
pip install -r requirements.txt 

### run
unset LD_LIBRARY_PATH
export WANDB_API_KEY=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

# QAM_EDIT
MUJOCO_GL=egl python main.py --run_group=reproduce --agent=agents/qam.py --tags=QAM_EDIT --seed=10001  --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True --agent.inv_temp=3.0 --agent.fql_alpha=0.0 --agent.edit_scale=0.1

# QAM_FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent=agents/qam.py --tags=QAM_FQL --seed=10001 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True --agent.inv_temp=10.0 --agent.fql_alpha=300.0 --agent.edit_scale=0.0 --online_steps=0

# QAM
MUJOCO_GL=egl python main.py --run_group=reproduce --agent=agents/qam.py --tags=QAM --seed=10001 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True --agent.inv_temp=3.0 --agent.fql_alpha=0.0 --agent.edit_scale=0.0 --online_steps=0

# DRIFT
MUJOCO_GL=egl python main.py \
  --run_group=reproduce \
  --agent=agents/drift.py \
  --tags=DRIFT \
  --seed=10001 \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --online_steps=0
