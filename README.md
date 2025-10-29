# Reinforcement Learning-Based Narrow Wall Traversal in Quadruped Robots

This repository is based on **Isaac Lab** and implements a **reinforcement learning (RL)** framework for training a quadruped robot (Unitree Go2) to traverse narrow corridors safely and efficiently.

---

## 🧠 Project Overview

The project aims to enable **autonomous narrow-wall traversal** using **reinforcement learning** techniques.  
It integrates **Isaac Sim**, **Isaac Lab**, and **RSL-RL** for parallel simulation and policy optimization.

---

## 📦 Tutorial

### 1. Clone Repository
```bash
git clone https://github.com/HuLixiaNick531/Reinforcement-Learning-Based-Narrow-Wall-Traversal-in-Quadruped-Robots.git
cd Reinforcement-Learning-Based-Narrow-Wall-Traversal-in-Quadruped-Robots
```

### 2. Install Dependencies
Make sure you have **Isaac Lab** and **RSL-RL** installed properly.  
Recommended environment:
- Python ≥ 3.10  
- NVIDIA GPU (for parallel simulation)
- Isaac Sim 4.x
- Isaac Lab (<= 2.2.0)

#### 2.1 create isaac env
If the IsaacLab environment is not created, use:
```bash
conda create -n env_isaaclab python=3.11
```
Where 'env_isaaclab' is the name of your own Isaac Lab environment.

To activate the Isaac Lab environment, using:
```bash
conda activate env_isaaclab 
```

In IsaacSim dict, using command to configure the IsaacSim env:
```bash
cd isaacsim
source ./setup_conda_env.sh
```

In IsaacLab dict, using command to create link with IsaacSim:
```bash
cd isaaclab
ln -s ~/software/isaacsim/ _isaac_sim
./isaaclab.sh --installcd
python -m pip install -e source/isaaclab_rl
```
'~/software/isaacsim/' should be replaced by the location of Isaac Sim.

#### 2.2 download dependencies for current project

To activate the Isaac Lab environment, using:
```bash
conda activate env_isaaclab 
```

To download dependencies, using:

```bash
cd Reinforcement-Learning-Based-Narrow-Wall-Traversal-in-Quadruped-Robots
python -m pip install -e source/unitree_gym  
```

<!-- cd unitree_gym   -->

<!-- Example (inside Isaac Lab environment):
```bash
conda activate env_isaaclab
python -m pip install -e source/robot_lab
``` -->
---

## 🚀 Training Command

First activate conda env:

```bash
conda activate env_isaaclab
source ~/software/isaacsim/setup_conda_env.sh 
```

'~/software/isaacsim/setup_conda_env.sh' should be replaced by 'setup_conda_env.sh' in isaacsim installation path.

To start training the quadruped agent, use commands below:

Without GUI (Recommanded):

```bash
python scripts/reinforcement_learning/rsl_rl/train.py     --task Isaac-Traverse-Walls-Unitree-Go2-v0     --num_envs 8 --headless
```

With GUI:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py     --task Isaac-Traverse-Walls-Unitree-Go2-v0     --num_envs 8
```

<!-- This launches RSL-RL with **400 parallel environments**, using the `Unitree Go2` robot model to learn a velocity-based navigation policy. -->

---


