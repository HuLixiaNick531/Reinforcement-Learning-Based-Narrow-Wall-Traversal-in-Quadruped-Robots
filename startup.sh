#!/bin/bash

# params
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

# create conda env
if conda env list | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
    echo "✅ Conda environment '${CONDA_ENV_NAME}' exists."
else
    echo "Conda environment '${CONDA_ENV_NAME}' not found. Try to create a new environment."
    conda create -n "${CONDA_ENV_NAME}" python=3.11 -y
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to create conda environment '${CONDA_ENV_NAME}'."
        exit 1
    fi
    echo "✅ Environment '${CONDA_ENV_NAME}' created successfully."
fi

# activate conda env
conda activate "${CONDA_ENV_NAME}"
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to activate conda environment '${CONDA_ENV_NAME}'."
    exit 1
fi
echo "✅ Environment '${CONDA_ENV_NAME}' activate successfully."

# activate isaac sim conda env
source "${ISAACSIM_LOCATION%/}"/setup_conda_env.sh
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to activate isaac sim conda environment in '${CONDA_ENV_NAME}'."
    exit 1
fi
echo "✅ Activate isaac sim conda environment in '${CONDA_ENV_NAME}' successfully."

# install isaac lab and rl package

LINK_PATH="${ISAACLAB_LOCATION%/}/_isaac_sim"
if [[ -L "$LINK_PATH" ]]; then
    TARGET=$(readlink "$LINK_PATH")
    echo "✅ Symbolic link exists: $LINK_PATH -> $TARGET"
else
    if [[ -e "$LINK_PATH" ]]; then
        echo "❌ '$LINK_PATH' exists but is NOT a symbolic link (maybe a folder or file). Please remove or rename it before proceeding."
        exit 1
    else
        echo "⚠️ Symbolic link not found: $LINK_PATH. Creating it now..."
        ln -s "${ISAACSIM_LOCATION}" "${LINK_PATH}"
        exit 1
    fi
fi

# install isaac lab
"${ISAACLAB_LOCATION%/}"/isaaclab.sh --install
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install Isaac Lab."
    exit 1
fi
echo "✅ Isaac Lab installed successfully."

# install isaaclab_rl
python -m pip install -e "${ISAACLAB_LOCATION%/}"/source/isaaclab_rl
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install isaaclab_rl package."
    exit 1
fi
echo "✅ isaaclab_rl package installed successfully."

# install unitree_gym
python -m pip install -e ${SCRIPT_PATH%/}/source/unitree_gym
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install unitree_gym package."
    exit 1
fi
echo "✅ unitree_gym package installed successfully."

# install toml
python -m pip install toml
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install toml package."
fi
echo "✅ toml package installed successfully."

# install own rsl_rl
echo "${SCRIPT_PATH%/}"
python -m pip install -e "${SCRIPT_PATH%/}"/3rd_party/rsl_rl
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install own rsl_rl package."
    exit 1
fi
echo "✅ own rsl_rl package installed successfully."

echo "✅ All setup steps completed successfully! You can now use the '${CONDA_ENV_NAME}' conda environment."

