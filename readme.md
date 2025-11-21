# Reinforcement-Learning-Based Narrow Wall Traversal in Quadruped Robots

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-3.0.1-green)](https://leggedrobotics.github.io/rsl-rl/)
[![Platform](https://img.shields.io/badge/platform-Linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

---

##  Overview

This repository provides training environments and reinforcement-learning pipelines for teaching **Unitree Go2** quadruped robots to traverse **extremely narrow corridors** using **contact-aware locomotion** within the Isaac Lab simulation framework.

The project includes:

- A custom Isaac Lab environment  
  **`Isaac-Traverse-Walls-Unitree-Go2-v0`**
- Curriculum-based corridor narrowing
- Contact-aware reward shaping
- RSL-RL PPO training support
- Tools for training, debugging, and replaying policies
- Example Omniverse extension integration

This project was developed for autonomous navigation in constrained spaces such as tunnels, collapsed corridors, and narrow industrial passages.

---

##  Installation

### 1. Install Isaac Lab  
Follow the official installation guide:

👉 https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

Conda installation is recommended for ease of use.

---

### 2. Clone this repository

```bash
git clone https://github.com/HuLixiaNick531/Reinforcement-Learning-Based-Narrow-Wall-Traversal-in-Quadruped-Robots.git
cd Reinforcement-Learning-Based-Narrow-Wall-Traversal-in-Quadruped-Robots
```

---

### 3. Install the environment

```bash
python -m pip install -e source/unitree_gym
```

---

## ⚠️ RSL-RL Version Requirement

This project requires **RSL-RL 3.0.1**.  
Please verify your installed version before starting training.

### 1. Check the installed version

```bash
pip show rsl-rl-lib
```

Expected output:

```
Name: rsl-rl-lib
Version: 3.0.1
```

### 2. If your version is not 3.0.1

Uninstall the existing version:

```bash
pip uninstall -y rsl-rl-lib
```

Install the correct version:

```bash
pip install rsl-rl-lib==3.0.1
```


##  Training

Training uses **RSL-RL** PPO implementation.

###  Start training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Traverse-Walls-Unitree-Go2-v0
```

Training logs and checkpoints are saved in:

```
logs/rsl_rl/unitree_go2_traverse/<timestamp>/
```

---

## ▶ Replay / Evaluate a trained policy

```bash
python3 scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Traverse-Walls-Unitree-Go2-v0 --load_run=2025-11-20_17-56-35 # Reinforcement-Learning-Based Narrow Wall Traversal in Quadruped Robots

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-3.0.1-green)](https://leggedrobotics.github.io/rsl-rl/)
[![Platform](https://img.shields.io/badge/platform-Linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

---

##  Overview

This repository provides training environments and reinforcement-learning pipelines for teaching **Unitree Go2** quadruped robots to traverse **extremely narrow corridors** using **contact-aware locomotion** within the Isaac Lab simulation framework.

The project includes:

- A custom Isaac Lab environment  
  **`Isaac-Traverse-Walls-Unitree-Go2-v0`**
- Curriculum-based corridor narrowing
- Contact-aware reward shaping
- RSL-RL PPO training support
- Tools for training, debugging, and replaying policies
- Example Omniverse extension integration

This project was developed for autonomous navigation in constrained spaces such as tunnels, collapsed corridors, and narrow industrial passages.

---

##  Installation

### 1. Install Isaac Lab  
Follow the official installation guide:

👉 https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

Conda installation is recommended for ease of use.

---

### 2. Clone this repository

```bash
git clone https://github.com/HuLixiaNick531/Reinforcement-Learning-Based-Narrow-Wall-Traversal-in-Quadruped-Robots.git
cd Reinforcement-Learning-Based-Narrow-Wall-Traversal-in-Quadruped-Robots
```

---

### 3. Install the environment

```bash
python -m pip install -e source/unitree_gym
```

---

## ⚠️ RSL-RL Version Requirement

This project requires **RSL-RL 3.0.1**.  
Please verify your installed version before starting training.

### 1. Check the installed version

```bash
pip show rsl-rl-lib
```

Expected output:

```
Name: rsl-rl-lib
Version: 3.0.1
```

### 2. If your version is not 3.0.1

Uninstall the existing version:

```bash
pip uninstall -y rsl-rl-lib
```

Install the correct version:

```bash
pip install rsl-rl-lib==3.0.1
```

### Why this matters?

The environment `Isaac-Traverse-Walls-Unitree-Go2-v0` depends on API behavior introduced in **RSL-RL 3.0.1**.  
Other versions (e.g., 2.x or 3.1.x) may cause training or playback failures.

---

##  Training

Training uses **RSL-RL** PPO implementation.

### 🔥 Start training

```bash
python3 scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Traverse-Walls-Unitree-Go2-v0
```

Training logs and checkpoints are saved in:

```
logs/rsl_rl/unitree_go2_traverse/<timestamp>/
```

---

## ▶ Replay / Evaluate a trained policy

```bash
python3 scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Traverse-Walls-Unitree-Go2-v0 --load_run=2025-11-21_15-27-57 --num_envs=32
```

---

## 🙌 Acknowledgements

- NVIDIA Isaac Sim & Isaac Lab  
- RSL-RL from ETH Zürich Legged Robotics  
- Unitree Go2 SDK + URDF models  
- NUS College of Design and Engineering
```

---

## 🙌 Acknowledgements

- NVIDIA Isaac Sim & Isaac Lab  
- RSL-RL from ETH Zürich Legged Robotics  
- Unitree Go2 SDK + URDF models  
- NUS College of Design and Engineering
