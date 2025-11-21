from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, quat_apply
from ..mdp.traverses import TraverseEvent 
from collections.abc import Sequence

if TYPE_CHECKING:
    from ..envs import TraverseManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 


# class reward_feet_edge(ManagerTermBase):
#     def __init__(self, cfg: RewardTermCfg, env: TraverseManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
#         self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
#         self.sensor_cfg = cfg.params["sensor_cfg"]
#         self.asset_cfg = cfg.params["asset_cfg"]
#         self.traverse_event: TraverseEvent = env.traverse_manager.get_term(cfg.params["traverse_name"])
#         self.body_id = self.contact_sensor.find_bodies('base')[0]
#         self.horizontal_scale = env.scene.terrain.cfg.terrain_generator.horizontal_scale
#         size_x, size_y = env.scene.terrain.cfg.terrain_generator.size
# #         self.rows_offset = (size_x * env.scene.terrain.cfg.terrain_generator.num_rows/2)
# #         self.cols_offset = (size_y * env.scene.terrain.cfg.terrain_generator.num_cols/2)
# #         total_x_edge_maskes = torch.from_numpy(self.traverse_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
# #         self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
# #             env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
# #         )

# #     def __call__(
# #         self,
# #         env: TraverseManagerBasedRLEnv,        
# #         asset_cfg: SceneEntityCfg,
# #         sensor_cfg: SceneEntityCfg,
# #         traverse_name: str,
# #         ) -> torch.Tensor:
# #         feet_pos_x = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,0] + self.rows_offset)
# #                       /self.horizontal_scale).round().long() 
# #         feet_pos_y = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,1] + self.cols_offset)
# #                       /self.horizontal_scale).round().long() 
# #         feet_pos_x = torch.clip(feet_pos_x, 0, self.x_edge_masks_tensor.shape[0]-1)
# #         feet_pos_y = torch.clip(feet_pos_y, 0, self.x_edge_masks_tensor.shape[1]-1)
# #         feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
# #         contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
# #         previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
# #         contact = torch.norm(contact_forces, dim=-1) > 2.
# #         last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
# #         contact_filt = torch.logical_or(contact, last_contacts) 
# #         self.feet_at_edge = contact_filt & feet_at_edge
# #         rew = (self.traverse_event.terrain.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
# #         # This is for debugging to matching index and x_edge_mask
# #         # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
# #         # cv2.imshow('origin',origin)
# #         # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
# #         # cv2.imshow('feet_edge',origin)
# #         # cv2.waitKey(1)
# #         return rew


# def reward_torques(
#     env: TraverseManagerBasedRLEnv,        
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor: 
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.sum(torch.square(asset.data.applied_torque), dim=1)


# def reward_dof_error(    
#     env: TraverseManagerBasedRLEnv,        
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor: 
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

# def reward_hip_pos(
#     env: TraverseManagerBasedRLEnv,        
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor: 
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
#                                     - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)


# def joint_pos_penalty(
#     env: TraverseManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg,
#     stand_still_scale: float,
#     velocity_threshold: float,
#     command_threshold: float,
# ) -> torch.Tensor:
#     """Penalize joint position error from default on the articulation."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
#     running_reward = torch.linalg.norm(
#         (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
#     )
#     reward = torch.where(
#         torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
#         running_reward,
#         stand_still_scale * running_reward,
#     )
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


# def feet_contact_stand_still(
#     env: TraverseManagerBasedRLEnv,
#     command_name: str,
#     sensor_cfg: SceneEntityCfg,
#     cmd_threshold: float = 0.1,
#     base_vel_threshold: float = 0.05,
#     contact_force_threshold: float = 1.0,
# ) -> torch.Tensor:
#     """Encourage keeping all feet in contact when no motion is commanded."""
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

#     # Combine the latest two contact readings to reduce flicker in the signal.
#     net_forces = contact_sensor.data.net_forces_w_history
#     contact_now = torch.norm(net_forces[:, 0, sensor_cfg.body_ids], dim=-1) > contact_force_threshold
#     contact_prev = torch.norm(net_forces[:, -1, sensor_cfg.body_ids], dim=-1) > contact_force_threshold
#     contact = torch.logical_or(contact_now, contact_prev)

#     # Reward reaches 1 when every tracked foot is in contact.
#     reward = torch.sum(contact, dim=-1).float() / contact.shape[-1]

#     # Use only the linear velocity component to decide if the robot should stand still;
#     # heading commands alone shouldn't disable the stand‑still reward.
#     cmd = env.command_manager.get_command(command_name)
#     cmd_mag = torch.abs(cmd[:, 0])
#     base_speed = torch.linalg.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)
#     stand_still_mask = torch.logical_and(cmd_mag < cmd_threshold, base_speed < base_vel_threshold)
#     reward *= stand_still_mask

#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


# def reward_ang_vel_xy(
#     env: TraverseManagerBasedRLEnv,        
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor: 
#     asset: Articulation = env.scene[asset_cfg.name]
#     reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: TraverseManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 用实际 action 维度，而不是 asset.num_joints
        act_term = env.action_manager.get_term("joint_pos")
        num_actions = act_term.raw_actions.shape[-1]

        # [num_envs, 2, num_actions]，存两帧动作
        self.previous_actions = torch.zeros(
            env.num_envs, 2, num_actions, dtype=torch.float, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """在 reset 的环境里把历史动作清零。"""
        if env_ids is None:
            # reset 全部环境
            self.previous_actions.zero_()
        else:
            self.previous_actions[env_ids, :, :] = 0.0

    def __call__(
        self,
        env: TraverseManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        # asset_cfg 未使用，仅为保持接口一致
        _ = asset_cfg

        # 当前动作
        cur_actions = env.action_manager.get_term("joint_pos").raw_actions

        # 滚动缓存：0 <- 旧的 1，1 <- 当前
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = cur_actions

        # 两帧动作差分的 L2 范数作为惩罚
        diff = self.previous_actions[:, 1, :] - self.previous_actions[:, 0, :]
        return torch.norm(diff, dim=1)


class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: TraverseManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_joint_vel = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """在 reset 的环境里把历史动作清零。"""
        if env_ids is None:
            # reset 全部环境
            self.previous_joint_vel.zero_()
        else:
            self.previous_joint_vel[env_ids, :, :] = 0.0

    def __call__(
        self,
        env: TraverseManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:,0,:]) / self.dt), dim=1)


def reward_common_upright(
    env: TraverseManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pitch/roll magnitude to encourage upright posture."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    reward = torch.abs(roll) + torch.abs(pitch)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def reward_common_smooth_action(
    env: TraverseManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize large raw actions for smoother commands."""
    _ = asset_cfg  # unused but kept for signature consistency
    actions = env.action_manager.get_term("joint_pos").raw_actions
    return torch.sum(torch.square(actions), dim=1)


def reward_common_joint_deviation(
    env: TraverseManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from default joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )


class reward_flat_dist(ManagerTermBase):
    """Progress reward: distance reduction toward the goal on flat stage."""

    def __init__(self, cfg: RewardTermCfg, env: TraverseManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.traverse_event: TraverseEvent = env.traverse_manager.get_term(cfg.params["traverse_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # Initialize previous distance to current distance to avoid spikes on first call.
        self.prev_dist = torch.norm(self.traverse_event.target_pos_rel, dim=-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        dist = torch.norm(self.traverse_event.target_pos_rel, dim=-1)
        if env_ids is None:
            self.prev_dist = dist
        else:
            self.prev_dist[env_ids] = dist[env_ids]

    def __call__(
        self,
        env: TraverseManagerBasedRLEnv,
        traverse_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        _ = traverse_name
        _ = asset_cfg
        dist = torch.norm(self.traverse_event.target_pos_rel, dim=-1)
        progress = self.prev_dist - dist
        progress = torch.clamp(progress, min=-0.2, max=0.2)
        self.prev_dist = dist
        flat_mask = (self.traverse_event.mode_flag > 0.5).float()
        return progress * flat_mask


def reward_flat_speed(
    env: TraverseManagerBasedRLEnv,
    traverse_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward speed along path direction on flat stage, scaled by uprightness."""
    traverse_event: TraverseEvent = env.traverse_manager.get_term(traverse_name)
    asset: Articulation = env.scene[asset_cfg.name]
    target_vec = traverse_event.target_pos_rel
    target_dir = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-5)
    vel = asset.data.root_vel_w[:, :2]
    forward_speed = torch.clamp(torch.sum(vel * target_dir, dim=-1), min=0.0)
    # Encourage staying upright while moving forward.
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    upright_factor = torch.exp(-2.0 * (torch.abs(roll) + torch.abs(pitch)))
    flat_mask = (traverse_event.mode_flag > 0.5).float()
    return forward_speed * upright_factor * flat_mask


def reward_flat_yaw(
    env: TraverseManagerBasedRLEnv,
    traverse_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw alignment with path direction on flat stage."""
    traverse_event: TraverseEvent = env.traverse_manager.get_term(traverse_name)
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    delta_yaw = wrap_to_pi(yaw - traverse_event.target_yaw)
    flat_mask = (traverse_event.mode_flag > 0.5).float()
    return torch.cos(delta_yaw) * flat_mask


def reward_flat_away_wall(
    env: TraverseManagerBasedRLEnv,
    traverse_name: str,
    sensor_cfg: SceneEntityCfg,
    fn_small: float = 1.0,
) -> torch.Tensor:
    """Discourage lateral pushing on walls after exiting (mode_flag == 1)."""
    traverse_event: TraverseEvent = env.traverse_manager.get_term(traverse_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    # Use lateral force magnitude as proxy for wall normal force.
    lateral_force = torch.norm(net_forces[..., :2], dim=-1)  # ignore vertical component
    excess = torch.clamp(torch.mean(lateral_force, dim=-1) - fn_small, min=0.0)
    flat_mask = (traverse_event.mode_flag > 0.5).float()
    return excess * flat_mask


def reward_wall_dist(
    env: TraverseManagerBasedRLEnv,
    traverse_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    desired_offset: float = 0.0,
) -> torch.Tensor:
    """Lateral distance penalty to path/wall on wall stage (mode_flag == 0)."""
    traverse_event: TraverseEvent = env.traverse_manager.get_term(traverse_name)
    asset: Articulation = env.scene[asset_cfg.name]
    # Gather current and previous goals to form a path segment.
    env_goals = traverse_event.env_goals
    max_idx = env_goals.shape[1] - 1
    cur_idx = torch.clamp(traverse_event.cur_goal_idx, 0, max_idx)
    prev_idx = torch.clamp(cur_idx - 1, 0, max_idx)
    cur_goal = env_goals.gather(1, cur_idx[:, None, None].expand(-1, -1, 3)).squeeze(1)
    prev_goal = env_goals.gather(1, prev_idx[:, None, None].expand(-1, -1, 3)).squeeze(1)
    path_vec = cur_goal[:, :2] - prev_goal[:, :2]
    path_dir = path_vec / (torch.norm(path_vec, dim=-1, keepdim=True) + 1e-5)
    robot_pos = asset.data.root_pos_w[:, :2] - traverse_event.env_origins[:, :2]
    rel_vec = robot_pos - prev_goal[:, :2]
    # Perpendicular distance from robot to the path line segment.
    lateral = torch.abs(rel_vec[:, 0] * path_dir[:, 1] - rel_vec[:, 1] * path_dir[:, 0])
    reward = torch.abs(lateral - desired_offset)
    wall_mask = (traverse_event.mode_flag <= 0.5).float()
    return reward * wall_mask


def reward_wall_speed(
    env: TraverseManagerBasedRLEnv,
    traverse_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward speed along path direction on wall stage, scaled by uprightness."""
    traverse_event: TraverseEvent = env.traverse_manager.get_term(traverse_name)
    asset: Articulation = env.scene[asset_cfg.name]
    target_vec = traverse_event.target_pos_rel
    target_dir = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-5)
    vel = asset.data.root_vel_w[:, :2]
    forward_speed = torch.clamp(torch.sum(vel * target_dir, dim=-1), min=0.0)
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    upright_factor = torch.exp(-2.0 * (torch.abs(roll) + torch.abs(pitch)))
    wall_mask = (traverse_event.mode_flag <= 0.5).float()
    return forward_speed * upright_factor * wall_mask


def reward_wall_yaw(
    env: TraverseManagerBasedRLEnv,
    traverse_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw alignment with path on wall stage."""
    traverse_event: TraverseEvent = env.traverse_manager.get_term(traverse_name)
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    delta_yaw = wrap_to_pi(yaw - traverse_event.target_yaw)
    wall_mask = (traverse_event.mode_flag <= 0.5).float()
    return torch.cos(delta_yaw) * wall_mask


def reward_wall_force(
    env: TraverseManagerBasedRLEnv,
    traverse_name: str,
    sensor_cfg: SceneEntityCfg,
    fn_target: float = 5.0,
) -> torch.Tensor:
    """Penalize deviation from desired lateral (wall-normal) force on wall stage."""
    traverse_event: TraverseEvent = env.traverse_manager.get_term(traverse_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    lateral_force = torch.norm(net_forces[..., :2], dim=-1)  # proxy for wall-normal
    avg_force = torch.mean(lateral_force, dim=-1)
    reward = torch.abs(avg_force - fn_target)
    wall_mask = (traverse_event.mode_flag <= 0.5).float()
    return reward * wall_mask


# def lin_vel_z_l2(env: TraverseManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize z-axis base linear velocity using L2 squared kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     reward = torch.square(asset.data.root_lin_vel_b[:, 2])
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


# def flat_orientation_l2(env: TraverseManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize non-flat base orientation using L2 squared kernel.

#     This is computed by penalizing the xy-components of the projected gravity vector.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


# def reward_feet_stumble(
#     env: TraverseManagerBasedRLEnv,        
#     sensor_cfg: SceneEntityCfg ,
#     ) -> torch.Tensor: 
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
#     rew = torch.any(torch.norm(net_contact_forces[:, :, :2], dim=2) >\
#             4 *torch.abs(net_contact_forces[:, :, 2]), dim=1)
#     return rew.float()


# def reward_tracking_goal_vel(
#     env: TraverseManagerBasedRLEnv, 
#     traverse_name: str, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     traverse_event: TraverseEvent = env.traverse_manager.get_term(traverse_name)
#     target_pos_rel = traverse_event.target_pos_rel
#     target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
#     cur_vel = asset.data.root_vel_w[:, :2]
#     proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
#     command_vel = env.command_manager.get_command('base_velocity')[:, 0]

#     # ignore near-zero commands to avoid dividing by ~0 and producing huge spikes
#     cmd_mask = command_vel > 0.05
#     command_vel = torch.clamp(command_vel, min=0.05)

#     # only reward forward motion along the target direction, cap to 1.0
#     proj_clamped = torch.clamp(proj_vel, min=0.0)
#     rew_move = proj_clamped / command_vel
#     rew_move = torch.clamp(rew_move, 0.0, 1.0)
#     rew_move = torch.where(cmd_mask, rew_move, torch.zeros_like(rew_move))
#     return rew_move


# def reward_tracking_yaw(     
#     env: TraverseManagerBasedRLEnv, 
#     traverse_name: str, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#     traverse_event: TraverseEvent =  env.traverse_manager.get_term(traverse_name)
#     asset: Articulation = env.scene[asset_cfg.name]
#     q = asset.data.root_quat_w
#     yaw = torch.atan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
#                     1 - 2*(q[:,2]**2 + q[:,3]**2))
#     return torch.exp(-torch.abs((traverse_event.target_yaw - yaw)))


# def reward_base_height(
#     env: TraverseManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     target_height: float = 0.27,
#     falloff: float = 0.06,
#     ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     base_height = asset.data.root_pos_w[:, 2]
#     # 改为正奖励：使用指数衰减函数，鼓励保持接近目标高度
#     # 当高度接近target_height时奖励接近1，远离时奖励衰减
#     height_error = torch.abs(base_height - target_height)
#     # 使用指数衰减，falloff控制衰减速度
#     rew = torch.exp(-height_error / (falloff + 1e-5))
#     return rew


# class reward_delta_torques(ManagerTermBase):
#     def __init__(self, cfg: RewardTermCfg, env: TraverseManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
#         self.previous_torque = torch.zeros(env.num_envs, 2,  self.asset.num_joints, dtype= torch.float ,device=self.device)

#     def reset(self, env_ids: Sequence[int] | None = None) -> None:
#         self.previous_torque[env_ids, 0,:] = 0.
#         self.previous_torque[env_ids, 1,:] = 0.

#     def __call__(
#         self,
#         env: TraverseManagerBasedRLEnv,        
#         asset_cfg: SceneEntityCfg,
#         ) -> torch.Tensor:
#         self.previous_torque[:, 0, :] = self.previous_torque[:, 1, :]
#         self.previous_torque[:, 1, :] = self.asset.data.applied_torque
#         return torch.sum(torch.square((self.previous_torque[:, 1, :] - self.previous_torque[:,0,:])), dim=1)


# def undesired_contacts(env: TraverseManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize undesired contacts as the number of violations that are above a threshold."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # check if contact force is above threshold
#     net_contact_forces = contact_sensor.data.net_forces_w_history
#     is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
#     # sum over contacts for each environment
#     reward = torch.sum(is_contact, dim=1).float()
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


# def reward_lin_vel_z(
#     env: TraverseManagerBasedRLEnv,        
#     traverse_name:str, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor: 
#     traverse_event: TraverseEvent =  env.traverse_manager.get_term(traverse_name)
#     terrain_names = traverse_event.env_per_terrain_name
#     asset: Articulation = env.scene[asset_cfg.name]
#     rew = torch.square(asset.data.root_lin_vel_b[:, 2])
#     rew[(terrain_names !='traverse_flat')[:,-1]] *= 0.5
#     return rew


# def reward_orientation(
#     env: TraverseManagerBasedRLEnv,   
#     traverse_name:str, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor: 
#     traverse_event: TraverseEvent =  env.traverse_manager.get_term(traverse_name)
#     terrain_names = traverse_event.env_per_terrain_name
#     asset: Articulation = env.scene[asset_cfg.name]
#     rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
#     rew[(terrain_names !='traverse_flat')[:,-1]] = 0.
#     return rew


# def reward_collision(
#     env: TraverseManagerBasedRLEnv, 
#     sensor_cfg: SceneEntityCfg ,
# ) -> torch.Tensor:
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
#     return torch.sum(1.*(torch.norm(net_contact_forces, dim=-1) > 0.1), dim=1)
