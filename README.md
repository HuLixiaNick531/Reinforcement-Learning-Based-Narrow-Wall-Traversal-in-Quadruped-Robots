conda activate env_isaaclab  /end cd unitree_gym /end
python -m pip install -e source/unitree_gym /emd 
python scripts/reinforcement_learning/rsl_rl/train.py     --task Isaac-Traverse-Walls-Unitree-Go2-v0     --num_envs=32
