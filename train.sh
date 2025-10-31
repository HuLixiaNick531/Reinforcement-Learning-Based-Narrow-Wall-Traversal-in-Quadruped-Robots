#!/bin/bash

# load params
source ./vars.sh

# check params
if [[ -z "$CONDA_ENV_NAME" ]]; then
    echo "❌ Error: CONDA_ENV_NAME is empty."
    echo "👉 Please set the CONDA_ENV_NAME variable before running this script."
    echo "   Example:"
    echo "     export CONDA_ENV_NAME=env_group22"
    exit 1
else
    echo "✅ CONDA_ENV_NAME = $CONDA_ENV_NAME"
fi
if [[ -z "$ISAACLAB_LOCATION" || -z "$ISAACSIM_LOCATION" ]]; then
    echo "❌ Error: ISAACLAB_LOCATION or ISAACSIM_LOCATION is empty."
    echo "👉 Please set both environment variables before running this script."
    echo "   Example:"
    echo "     export ISAACLAB_LOCATION=/path/to/isaaclab"
    echo "     export ISAACSIM_LOCATION=/path/to/isaac-sim"
    exit 1
else
    echo "✅ ISAACLAB_LOCATION = $ISAACLAB_LOCATION"
    echo "✅ ISAACSIM_LOCATION = $ISAACSIM_LOCATION"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}"
echo "✅ SCRIPT_PATH = $SCRIPT_PATH"

# add conda hook
eval "$(conda shell.bash hook)"

# activate conda env
conda activate "${CONDA_ENV_NAME}"
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to activate conda environment '${CONDA_ENV_NAME}'."
    exit 1
fi
echo "✅ Environment '${CONDA_ENV_NAME}' activate successfully."

# change actor critic network files
if [[ -f "${SCRIPT_PATH%/}/scripts/reinforcement_learning/rsl_rl/actor_critic.py" ]]; then
    cp "${SCRIPT_PATH%/}/scripts/reinforcement_learning/rsl_rl/actor_critic.py" "${SCRIPT_PATH%/}/3rd_party/rsl_rl/rsl_rl/modules/actor_critic.py"
    echo "✅ Actor-critic network file found."
else
    echo "⚠️ Actor-critic network file not found, use the original one."
    if [[ -f "${SCRIPT_PATH%/}/3rd_party/rsl_rl/rsl_rl/modules/actor_critic.py.bak" ]]; then
        cp "${SCRIPT_PATH%/}/3rd_party/rsl_rl/rsl_rl/modules/actor_critic.py.bak" "${SCRIPT_PATH%/}/3rd_party/rsl_rl/rsl_rl/modules/actor_critic.py"
    fi
fi

# activate isaac sim conda env
source "${ISAACSIM_LOCATION%/}"/setup_conda_env.sh
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to activate isaac sim conda environment in '${CONDA_ENV_NAME}'."
    exit 1
fi
echo "✅ Activate isaac sim conda environment in '${CONDA_ENV_NAME}' successfully."

# run training script
python ${SCRIPT_PATH%/}/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Traverse-Walls-Unitree-Go2-v0 --num_envs 4 --headless