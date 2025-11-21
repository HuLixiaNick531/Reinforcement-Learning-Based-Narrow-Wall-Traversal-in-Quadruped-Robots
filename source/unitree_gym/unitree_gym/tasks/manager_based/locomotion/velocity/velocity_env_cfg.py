# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import math
import sys
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, RayCasterCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from ..terrains.traverse_terrain_importer import TraverseTerrainImporter
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp.events import ( 
randomize_rigid_body_mass,
apply_external_force_torque,
reset_joints_by_scale
)

import unitree_gym.tasks.manager_based.locomotion.velocity.mdp as mdp

from .mdp import terminations, rewards, traverses, events, observations, traverse_commands
##
# Pre-defined configs
##
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from unitree_gym.tasks.manager_based.locomotion.terrains.extreme_traverse.config.traverse import TRAVERSE_TERRAINS_CFG
# from unitree_gym.tasks.manager_based.locomotion.velocity.terrains.wall import WALLS_TERRAINS_CFG
from .envs import TraverseManagerBasedRLEnvCfg
from .mdp.traverse_actions import DelayedJointPositionActionCfg 
# TraverseEventsCfg, TeacherRewardsCfg....
##
# Scene definition
##

@configclass
class MySceneGo2Cfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        class_type=TraverseTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TRAVERSE_TERRAINS_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", 
                                      history_length=2, 
                                      track_air_time=True, 
                                      debug_vis= False,
                                      force_threshold=1.
                                      )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = traverse_commands.TraverseCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0,6.0 ),
        heading_control_stiffness=0.8,
        ranges=traverse_commands.TraverseCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.3), 
            heading=(-1.6, 1.6)
        ),
        clips= traverse_commands.TraverseCommandCfg.Clips(
            lin_vel_clip = 0.2,
            ang_vel_clip = 0.4
        )
    )

@configclass
class TraverseEventsCfg:
    """Command specifications for the MDP."""
    base_traverse = traverses.TraverseEventsCfg(
        asset_name = 'robot',
        next_goal_threshold=0.0,
        )


@configclass
class ActionsCfg:
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True,
        action_delay_steps = [1, 1],
        delay_update_global_steps = 24 * 8000,
        history_length = 8,
        use_delay = True,
        clip = {'.*': (-4.8,4.8)}
        )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        policy_traverse_observations = ObsTerm(
            func=observations.TraversePolicyObservations,
            params={            
            "asset_cfg":SceneEntityCfg("robot"),
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "traverse_name":'base_traverse',
            "history_length": 10
            },
            clip= (-100,100)
        )
    policy: PolicyCfg = PolicyCfg()

    # @configclass
    # class CriticCfg(ObsGroup):
    #     """Observations for policy group."""
    #     # observation terms (order preserved)
    #     critic_traverse_observations = ObsTerm(
    #         func=observations.CriticTraverseObservations,
    #         params={            
    #         "asset_cfg":SceneEntityCfg("robot"),
    #         "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "traverse_name":'base_traverse',
    #         "history_length": 10
    #         },
    #         clip= (-100,100)
    #     )
    # # observation groups
    # critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    ### Modified origin events, plz see relative issue https://github.com/isaac-sim/IsaacLab/issues/1955
    """Configuration for events."""
    reset_root_state = EventTerm(
        func= events.reset_root_state,
        params = {'offset': 3.5},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func= reset_joints_by_scale, 
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
        mode="reset",
    )
    physics_material = EventTerm( # Okay
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.6, 2.0),
            "num_buckets": 64,
        },
    )

    ## we don't use this event, If you use this, you will get a bad result
    # randomize_actuator_gains = EventTerm(
    #     func= events.randomize_actuator_gains,
    #     params={
    #         "asset_cfg" :SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.975, 1.025),  
    #         "damping_distribution_params": (0.975, 1.025),
    #         "operation": "scale",
    #         },
    #     mode="startup",
    # )
    randomize_rigid_body_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1., 3.0),
            "operation": "add",
            },
    )
    randomize_rigid_body_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {'x':(-0.02, 0.02),'y':(-0.02, 0.02),'z':(-0.02, 0.02)}
            },
    )
    # random_camera_position = EventTerm(
    #     func= events.random_camera_position,
    #     mode="startup",
    #     params={'sensor_cfg':SceneEntityCfg("depth_camera"),
    #             'rot_noise_range': {'pitch':(-5, 5)},
    #             'convention':'ros',
    #             },
    # )
    push_by_setting_velocity = EventTerm( # Okay
        func = events.push_by_setting_velocity, 
        params={'velocity_range':{"x":(-0.25, 0.25), "y":(-0.25, 0.25)}},
        interval_range_s = (12. ,16. ),
        is_global_time= True, 
        mode="interval",
    )
    base_external_force_torque = EventTerm(  # Okay
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

# # Available Body strings:
#     reward_collision = RewTerm(
#         func=rewards.undesired_contacts, 
#         weight=-0.0,  # 降低权重：从-10.0改为-5.0，避免惩罚过重导致策略过于保守
#         params={
#             "sensor_cfg":SceneEntityCfg("contact_forces", body_names=["base",".*_calf",".*_thigh"]),
#         },
#     )
#     reward_hip_pos = RewTerm(
#         func=rewards.reward_hip_pos, 
#         weight=-0.5,
#         params={
#             "asset_cfg":SceneEntityCfg("robot", joint_names=".*_hip_joint"),
#         },
#     )
#     # reward_feet_edge = RewTerm(
#     #     func=rewards.reward_feet_edge, 
#     #     weight=-1.0,
#     #     params={
#     #         "asset_cfg":SceneEntityCfg(name="robot", body_names=["FL_foot","FR_foot","RL_foot","RR_foot"]),
#     #         "sensor_cfg":SceneEntityCfg(name="contact_forces", body_names=".*_foot"),
#     #         "traverse_name":'base_traverse',
#     #     },
#     # )
#     reward_torques = RewTerm(
#         func=rewards.reward_torques, 
#         weight=-2.5e-4,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#         },
#     )
#     # reward_dof_error = RewTerm(
#     #     func=rewards.reward_dof_error, 
#     #     weight=-0.04,
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot"),
#     #     },
#     # )
#     joint_pos_penalty = RewTerm(
#         func=mdp.joint_pos_penalty,
#         weight=-1.0,
#         params={
#             "command_name": "base_velocity",
#             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
#             "stand_still_scale": 5.0,
#             "velocity_threshold": 0.5,
#             "command_threshold": 0.1,
#         },
#     )
#     reward_ang_vel_xy = RewTerm(
#         func=rewards.reward_ang_vel_xy, 
#         weight=-0.05,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#         },
#     )
#     reward_action_rate = RewTerm(
#         func=rewards.reward_action_rate, 
#         weight=-0.1,
#         params={
#           "asset_cfg": SceneEntityCfg("robot"),
#         },
#     )
#     reward_dof_acc = RewTerm(
#         func=rewards.reward_dof_acc, 
#         weight=-2.5e-7,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#         },
#     )
#     lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
#     flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
#     reward_base_height = RewTerm(
#         func=rewards.reward_base_height,
#         weight=5.0,  # 大幅提高权重：强烈鼓励保持站立高度，防止机器人向前摔倒
#         params={
#             "asset_cfg":SceneEntityCfg("robot"),
#             "target_height":0.27,
#             "falloff":0.06,
#         },
#     )
#     # reward_feet_stumble = RewTerm(
#     #     func=rewards.reward_feet_stumble, 
#     #     weight=-1.0,
#     #     params={
#     #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
#     #     },
#     # )

#     reward_tracking_goal_vel = RewTerm(
#         func=rewards.reward_tracking_goal_vel, 
#         weight=1.0,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#             "traverse_name": 'base_traverse'
#         },
#     )
#     reward_tracking_yaw = RewTerm(
#         func=rewards.reward_tracking_yaw, 
#         weight=0.5,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#             "traverse_name": 'base_traverse'
#         },
#     )
#     reward_delta_torques = RewTerm(
#         func=rewards.reward_delta_torques, 
#         weight=-1.0e-7,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#         },
#     )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    ['base', 
    'FL_hip', 
    'FL_thigh', 
    'FL_calf', 
    'FL_foot', 
    'FR_hip', 
    'FR_thigh', 
    'FR_calf', 
    'FR_foot', 
    'Head_upper', 
    'Head_lower', 
    'RL_hip', 
    'RL_thigh', 
    'RL_calf', 
    'RL_foot', 
    'RR_hip', 
    'RR_thigh', 
    'RR_calf',
    'RR_foot']
    """
# Available Body strings: 
    reward_common_upright = RewTerm(
        func=rewards.reward_common_upright,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_common_smooth_action = RewTerm(
        func=rewards.reward_common_smooth_action,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_common_joint_deviation = RewTerm(
        func=rewards.reward_common_joint_deviation,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_flat_dist = RewTerm(
        func=rewards.reward_flat_dist,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "traverse_name": "base_traverse",
        },
    )
    reward_flat_speed = RewTerm(
        func=rewards.reward_flat_speed,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "traverse_name": "base_traverse",
        },
    )
    reward_flat_yaw = RewTerm(
        func=rewards.reward_flat_yaw,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "traverse_name": "base_traverse",
        },
    )
    reward_flat_away_wall = RewTerm(
        func=rewards.reward_flat_away_wall,
        weight=-0.5,
        params={
            "traverse_name": "base_traverse",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "fn_small": 1.0,
        },
    )
    reward_wall_dist = RewTerm(
        func=rewards.reward_wall_dist,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "traverse_name": "base_traverse",
            "desired_offset": 0.0,
        },
    )
    reward_wall_speed = RewTerm(
        func=rewards.reward_wall_speed,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "traverse_name": "base_traverse",
        },
    )
    reward_wall_yaw = RewTerm(
        func=rewards.reward_wall_yaw,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "traverse_name": "base_traverse",
        },
    )
    reward_wall_force = RewTerm(
        func=rewards.reward_wall_force,
        weight=-0.5,
        params={
            "traverse_name": "base_traverse",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "fn_target": 5.0,
        },
    )



    # reward_collision = RewTerm(
    #     func=rewards.reward_collision, 
    #     weight=-10., 
    #     params={
    #         "sensor_cfg":SceneEntityCfg("contact_forces", body_names=["base",".*_calf",".*_thigh"]),
    #     },
    # )
    # reward_torques = RewTerm(
    #     func=rewards.reward_torques, 
    #     weight=-0.00001, 
    #     params={
    #         "asset_cfg":SceneEntityCfg("robot"),
    #     },
    # )
    # reward_dof_error = RewTerm(
    #     func=rewards.reward_dof_error, 
    #     weight=-0.04, 
    #     params={
    #         "asset_cfg":SceneEntityCfg("robot"),
    #     },
    # )
    # reward_hip_pos = RewTerm(
    #     func=rewards.reward_hip_pos, 
    #     weight=-0.5, 
    #     params={
    #         "asset_cfg":SceneEntityCfg("robot", joint_names=".*_hip_joint"),
    #     },
    # )
    # reward_ang_vel_xy = RewTerm(
    #     func=rewards.reward_ang_vel_xy, 
    #     weight=-0.3, 
    #     params={
    #         "asset_cfg":SceneEntityCfg("robot"),
    #     },
    # )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate, 
        weight=-0.3,
        params={
          "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc, 
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # reward_lin_vel_z = RewTerm(
    #     func=rewards.reward_lin_vel_z, 
    #     weight=-1.0, 
    #     params={
    #         "asset_cfg":SceneEntityCfg("robot"),
    #         "traverse_name":'base_traverse',
    #     },
    # )
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # reward_tracking_goal_vel = RewTerm(
    #     func=rewards.reward_tracking_goal_vel, 
    #     weight=15.0,
    #     params={
    #         "asset_cfg":SceneEntityCfg("robot"),
    #         "traverse_name":'base_traverse'
    #     },
    # )
    # reward_feet_contact_stand_still = RewTerm(
    #     func=rewards.feet_contact_stand_still,
    #     weight=0.2,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )
    # reward_tracking_yaw = RewTerm(
    #     func=rewards.reward_tracking_yaw, 
    #     weight=10, 
    #     params={
    #         "asset_cfg":SceneEntityCfg("robot"),
    #         "traverse_name":'base_traverse'
    #     },
    # )
    # reward_delta_torques = RewTerm(
    #     func=rewards.reward_delta_torques, 
    #     weight=-1.0e-7, 
    #     params={
    #         "asset_cfg":SceneEntityCfg("robot"),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    total_terminates = DoneTerm(
        func=terminations.terminate_episode, 
        time_out=True,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )


# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

#     command_levels = CurrTerm(
#         func=mdp.command_levels_vel,
#         params={
#             "reward_term_name": "track_lin_vel_xy_exp",
#             "range_multiplier": (0.1, 1.0),
#         },
#     )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # command_levels = CurrTerm(
    #     func=mdp.command_levels_vel,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "range_multiplier": (0.1, 1.0),
    #     },
    # )

    # wall_width_curriculum = CurrTerm(
    #     func=mdp.wall_curriculum_update, # <--- 调用我们定义的函数
    #     params={
    #         "initial_width": 0.6,  # 初始墙壁间距（例如 0.7米）
    #         "final_width": 0.4,   # 最终墙壁间距（例如 0.4米）
    #         "max_round": 20000,  # 课程持续的最大回合数
    #     },
    # )
    # # --------------------------


##
# Environment configuration
##


@configclass
class LocomotionVelocityGo2EnvCfg(TraverseManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneGo2Cfg = MySceneGo2Cfg(num_envs=1024, env_spacing=6.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    traverses: TraverseEventsCfg = TraverseEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)


def create_obsgroup_class(class_name, terms, enable_corruption=False, concatenate_terms=True):
    """
    Dynamically create and register a ObsGroup class based on the given configuration terms.

    :param class_name: Name of the configuration class.
    :param terms: Configuration terms, a dictionary where keys are term names and values are term content.
    :param enable_corruption: Whether to enable corruption for the observation group. Defaults to False.
    :param concatenate_terms: Whether to concatenate the observation terms in the group. Defaults to True.
    :return: The dynamically created class.
    """
    # Dynamically determine the module name
    module_name = inspect.getmodule(inspect.currentframe()).__name__

    # Define the post-init function
    def post_init_wrapper(self):
        setattr(self, "enable_corruption", enable_corruption)
        setattr(self, "concatenate_terms", concatenate_terms)

    # Dynamically create the class using ObsGroup as the base class
    terms["__post_init__"] = post_init_wrapper
    dynamic_class = configclass(type(class_name, (ObsGroup,), terms))

    # Custom serialization and deserialization
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # Add custom serialization methods to the class
    dynamic_class.__getstate__ = __getstate__
    dynamic_class.__setstate__ = __setstate__

    # Place the class in the global namespace for accessibility
    globals()[class_name] = dynamic_class

    # Register the dynamic class in the module's dictionary
    if module_name in sys.modules:
        sys.modules[module_name].__dict__[class_name] = dynamic_class
    else:
        raise ImportError(f"Module {module_name} not found.")

    # Return the class for external instantiation
    return dynamic_class
