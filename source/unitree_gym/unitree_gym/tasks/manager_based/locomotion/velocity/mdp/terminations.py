# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from ..mdp import TraverseEvent

if TYPE_CHECKING:
    from ..envs import TraverseManagerBasedRLEnv


def terminate_episode(
    env: TraverseManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):  
    reset_buf = torch.zeros((env.num_envs, ), dtype=torch.bool, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:,3:7])
    # tighten attitude thresholds so impending falls end the episode sooner
    roll_cutoff = torch.abs(wrap_to_pi(roll)) > 1.0
    pitch_cutoff = torch.abs(wrap_to_pi(pitch)) > 1.0
    time_out_buf = env.episode_length_buf >= env.max_episode_length
    traverse_event: TraverseEvent =  env.traverse_manager.get_term('base_traverse')    
    reach_goal_cutoff = traverse_event.cur_goal_idx >= env.scene.terrain.cfg.terrain_generator.num_goals
    # treat base heights close to ground as failure to prevent crawling
    # 放宽高度阈值：从0.13提高到0.12，给机器人更多学习时间，避免轻微前倾就终止
    height_cutoff = asset.data.root_state_w[:, 2] < 0.12
    time_out_buf |= reach_goal_cutoff
    reset_buf |= time_out_buf
    reset_buf |= roll_cutoff
    reset_buf |= pitch_cutoff
    reset_buf |= height_cutoff
    return reset_buf
