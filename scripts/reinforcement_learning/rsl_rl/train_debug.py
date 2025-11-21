# /home/pp/unitree_gym（复件）/scripts/reinforcement_learning/rsl_rl/train_debug.py
import os, sys

# 1) 把“项目根目录”加入 sys.path（就是包含 scripts/ 的那一层）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# 结果应当是 /home/pp/unitree_gym（复件）
sys.path.insert(0, ROOT)

# # 2) 现在就能当成包导入了
# from scripts.reinforcement_learning.rsl_rl import train

# 3) 想模拟命令行参数就重写 sys.argv
# sys.argv = [
#     "train.py",
#     "--task=RobotLab-Isaac-Velocity-Rough-Hexapod-v0",
#     "--headless",
#     "--num_envs=64",
#     "--max_iterations=2000",
# ]

# sys.argv = [
#     "train.py",
#     "--task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-Lite3-v0",
#     "--headless",
#     "--num_envs=64",
#     "--max_iterations=10000",
# ]

sys.argv = [
    "train.py",
    "--task=Isaac-Traverse-Walls-Unitree-Go2-v0",
    "--logger=wandb",
    "--log_project_name=go2",
    # "--headless",
    "--num_envs=512",
    "--max_iterations=10000",
]


# sys.argv = [
#     "play.py",
#     "--task=Isaac-Traverse-Walls-Unitree-Go2-v0",
#     "--headless",
#     "--num_envs=4",
#     "--load_run=2025-11-20_17-56-35",
# ]




from scripts.reinforcement_learning.rsl_rl import train
# from scripts.reinforcement_learning.rsl_rl import play

if __name__ == "__main__":
    train.main()
    # play.main()