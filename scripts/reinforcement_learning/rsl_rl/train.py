# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# 使用cli_args函数给parser加上强化学习的参数
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# args_cli：存放 Isaac/argparse 自己的参数
# hydra_args：保留 Hydra 专用参数，稍后再交给 Hydra 处理

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app/ isaac sim 启动
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_gym.tasks  # noqa: F401

from custom_actor_critic import CustomActorCritic

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

VISUALIZATION = False

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if VISUALIZATION:
        print("\n===== Environment Space Info =====")
        print("Observation space:", env.observation_space)
        if hasattr(env.observation_space, "shape"):
            print("Observation shape:", env.observation_space.shape)

        print("Action space:", env.action_space)
        if hasattr(env.action_space, "shape"):
            print("Action dim:", env.action_space.shape)
        if hasattr(env.action_space, "low"):
            print("Action low:", env.action_space.low)
        if hasattr(env.action_space, "high"):
            print("Action high:", env.action_space.high)
        print("==================================\n")

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    custom_ac = CustomActorCritic(env).to(agent_cfg.device)
    if hasattr(runner, "algo") and hasattr(runner.algo, "actor_critic"):
        runner.algo.actor_critic = custom_ac
    if hasattr(runner, "actor_critic"):
        runner.actor_critic = custom_ac

    if VISUALIZATION:
        import torch
        import torch.nn as nn
        from torch.utils.tensorboard import SummaryWriter

        class _ACGraphWrapper(nn.Module):
            """Wrap CustomActorCritic so add_graph can trace (mu, V) from (policy_obs, critic_obs)."""
            def __init__(self, ac: nn.Module):
                super().__init__()
                self.ac = ac

            def forward(self, pol_obs: torch.Tensor, cri_obs: torch.Tensor):
                # Adjust these lines if your CustomActorCritic uses different attribute names
                feat_p = self.ac.policy_encoder(pol_obs)
                feat_c = self.ac.critic_encoder(cri_obs)
                h = self.ac.actor.backbone(feat_p)      # e.g., MLP trunk inside DiagGaussianActor
                mu = self.ac.actor.mu_head(h)           # policy mean
                V  = self.ac.critic(feat_c)             # state value
                return mu, V

        # Build dummy inputs matching env.observation_space exactly.
        def _space_dummy(space, device):
            assert isinstance(space, gym.spaces.Box), "Only Box observation spaces are supported here."
            shape = space.shape
            return torch.zeros((1, *shape), dtype=torch.float32, device=device)

        if isinstance(env.observation_space, gym.spaces.Dict):
            pol_space = env.observation_space["policy"]
            cri_space = env.observation_space["critic"]
            pol_dummy = _space_dummy(pol_space, agent_cfg.device)
            cri_dummy = _space_dummy(cri_space, agent_cfg.device)
        else:
            # Single-branch obs goes to both policy and critic
            base_space = env.observation_space
            pol_dummy = _space_dummy(base_space, agent_cfg.device)
            cri_dummy = _space_dummy(base_space, agent_cfg.device)

        tb_dir = os.path.join(log_dir, "tb_graph")
        os.makedirs(tb_dir, exist_ok=True)

        wrapper = _ACGraphWrapper(custom_ac).to(agent_cfg.device).eval()

        try:
            with SummaryWriter(tb_dir) as w:
                # verbose=False avoids large console logs
                w.add_graph(wrapper, (pol_dummy, cri_dummy), verbose=False)
            print(f"[INFO] TensorBoard graph written to: {tb_dir}")
            print("       Run: tensorboard --logdir", log_dir)
        except Exception as e:
            print(f"[WARN] add_graph failed: {e}")

        onnx_path = os.path.join(log_dir, "ac_model_graph.onnx")
        torch.onnx.export(
            _ACGraphWrapper(custom_ac).to(agent_cfg.device).eval(),
            (pol_dummy, cri_dummy),
            onnx_path,
            input_names=["policy_obs","critic_obs"],
            output_names=["mu","V"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={"policy_obs":{0:"batch"},"critic_obs":{0:"batch"},
                        "mu":{0:"batch"},"V":{0:"batch"}}
        )
        print("[INFO] Exported ONNX to:", onnx_path)

    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
