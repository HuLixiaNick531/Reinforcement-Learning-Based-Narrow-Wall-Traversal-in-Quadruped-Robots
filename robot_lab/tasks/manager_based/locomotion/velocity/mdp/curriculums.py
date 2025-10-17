# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def wall_curriculum_update(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    initial_width: float,
    final_width: float,
    max_round: float,
) -> None:
    
    # Check if walls exist
    if "wall_left" not in env.scene.rigid_objects or "wall_right" not in env.scene.rigid_objects:
        return
    
    # Record robot initial positions (first call only)
    if not hasattr(env, '_robot_initial_positions'):
        robot = env.scene["robot"]
        env._robot_initial_positions = robot.data.root_pos_w.clone()  # [num_envs, 3]
        print(f"Recorded {env.num_envs} robot initial positions")
    
    # Only update when environments reset (robot death)
    if len(env_ids) == 0:
        return
    
    # Initialize death counter (first call only)
    if not hasattr(env, '_wall_curriculum_death_count'):
        env._wall_curriculum_death_count = 0
    
    # Increment death count
    env._wall_curriculum_death_count += len(env_ids)
    
    # Calculate progress (0.0 to 1.0)
    progress = min(env._wall_curriculum_death_count / max_round, 1.0)
    
    # Calculate current width
    current_width = initial_width - (initial_width - final_width) * progress
    
    wall_left_entity = env.scene["wall_left"]
    wall_right_entity = env.scene["wall_right"]
    
    # Convert env_ids to tensor
    if isinstance(env_ids, (list, tuple)):
        reset_env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        reset_env_ids = env_ids
    
    # Get current wall positions
    left_positions = wall_left_entity.data.root_pos_w.clone()
    right_positions = wall_right_entity.data.root_pos_w.clone()
    
    # Update walls based on robot initial positions
    for env_id in reset_env_ids:
        robot_initial_pos = env._robot_initial_positions[env_id]  # [3]
        
        # X: robot initial position + 2m forward
        wall_x = robot_initial_pos[0] + 2.0
        
        # Y: robot initial Y ± width/2
        left_y = robot_initial_pos[1] + current_width / 2.0   # left wall
        right_y = robot_initial_pos[1] - current_width / 2.0  # right wall
        
        # Z: fixed height
        wall_z = 0.6
        
        # Update wall positions
        left_positions[env_id, 0] = wall_x
        left_positions[env_id, 1] = left_y
        left_positions[env_id, 2] = wall_z
        
        right_positions[env_id, 0] = wall_x
        right_positions[env_id, 1] = right_y
        right_positions[env_id, 2] = wall_z
    
    # Update wall positions
    wall_left_entity.data.root_pos_w[:] = left_positions
    wall_right_entity.data.root_pos_w[:] = right_positions
    
    # Write to simulation (reset environments only)
    left_pos_reset = left_positions[reset_env_ids]
    right_pos_reset = right_positions[reset_env_ids]
    left_quat_reset = wall_left_entity.data.root_quat_w[reset_env_ids]
    right_quat_reset = wall_right_entity.data.root_quat_w[reset_env_ids]
    
    left_pose_reset = torch.cat([left_pos_reset, left_quat_reset], dim=-1)
    right_pose_reset = torch.cat([right_pos_reset, right_quat_reset], dim=-1)
    
    wall_left_entity.write_root_pose_to_sim(left_pose_reset, reset_env_ids)
    wall_right_entity.write_root_pose_to_sim(right_pose_reset, reset_env_ids)
    
    # Debug output
    print(f"Deaths: {env._wall_curriculum_death_count}, Reset envs: {len(env_ids)}, Width: {current_width:.3f}m, Progress: {progress:.1%}")
    
    return current_width



def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)
