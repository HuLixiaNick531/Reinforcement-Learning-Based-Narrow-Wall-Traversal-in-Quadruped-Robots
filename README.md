# Reinforcement Learning-Based Narrow Wall Traversal in Quadruped Robots

This repository is based on **Isaac Lab** and implements a **reinforcement learning (RL)** framework for training a quadruped robot (Unitree Go2) to traverse narrow corridors safely and efficiently.

---

## 🧠 Project Overview

The project aims to enable **autonomous narrow-wall traversal** using **reinforcement learning** techniques.  
It integrates **Isaac Sim**, **Isaac Lab**, and **RSL-RL** for parallel simulation and policy optimization.

---

## 📦 Environment Setup

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
- Isaac Lab (latest release)

Example (inside Isaac Lab environment):
```bash
conda activate env_isaaclab
python -m pip install -e source/robot_lab
```
---

## 🚀 Training Command

To start training the quadruped agent:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py     --task RobotLab-Isaac-Velocity-Narrow-Unitree-Go2-v0     --num_envs=400
```

This launches RSL-RL with **400 parallel environments**, using the `Unitree Go2` robot model to learn a velocity-based navigation policy.

---


## 🧩 Repository Structure

The project follows the Isaac Lab extension layout, with modularized folders for robot assets, tasks, and training logic.

```
.
├── config/                                # Isaac Lab extension configuration
│   └── extension.toml
│
├── data/                                  # Robot model and mesh resources
│   └── Robots/
│       └── unitree/
│           └── go2_description/
│               ├── meshes/                # Mesh files for Go2 robot (DAE format)
│               │   ├── base.dae
│               │   ├── calf.dae
│               │   ├── foot.dae
│               │   └── ...
│               └── urdf/
│                   └── go2_description.urdf
│
├── pyproject.toml                         # Project metadata and build config
├── setup.py                               # Setup script for installation
│
├── robot_lab/                             # Core Isaac Lab extension source
│   ├── assets/                            # Robot asset definitions
│   │   ├── backup_origin/
│   │   │   └── unitree.py
│   │   └── unitree.py
│   │
│   ├── tasks/                             # Task and environment definitions
│   │   ├── direct/                        # Direct control mode
│   │   └── manager_based/                 # Manager-based task definition
│   │       └── locomotion/
│   │           └── velocity/
│   │               ├── config/            # Configuration for quadruped tasks
│   │               │   └── quadruped/unitree_go2/
│   │               ├── mdp/               # MDP components (commands, rewards, etc.)
│   │               │   ├── commands.py
│   │               │   ├── rewards.py
│   │               │   └── curriculums.py
│   │               └── velocity_env_cfg.py
│   │
│   └── ui_extension_example.py            # Optional Isaac Lab UI extension example
│
├── robot_lab.egg-info/                    # Build artifacts (auto-generated)
│   ├── PKG-INFO
│   └── SOURCES.txt
│
└── scripts/
    └── reinforcement_learning/
        └── rsl_rl/
            ├── train.py                   # Main training script (entry point)
            ├── configs/                   # PPO and task hyperparameters
            └── utils/                     # (Optional) helper tools
```



---

### 🔍 Key Components

| Folder | Description |
|--------|--------------|
| `data/Robots/unitree/` | Stores robot models, meshes, and URDFs. |
| `robot_lab/assets/` | Python definitions for robot assets used in simulation. |
| `robot_lab/tasks/manager_based/` | Main RL environments built on Isaac Lab's task manager. |
| `robot_lab/tasks/manager_based/locomotion/velocity/` | Contains velocity-control environments and curriculum logic. |
| `scripts/reinforcement_learning/rsl_rl/train.py` | Main script to launch reinforcement learning training. |


---

## 📊 Results & Visualization


---

## 🙏 Acknowledgements

This project draws inspiration and partial structure from the excellent open-source repository:

- [**fan-ziqi/robot_lab**](https://github.com/fan-ziqi/robot_lab)

We sincerely thank the original authors for their open-source contributions.  
Their work on **Isaac Lab-based robot simulation and reinforcement learning framework** provided valuable reference and a solid foundation for this project’s development.

