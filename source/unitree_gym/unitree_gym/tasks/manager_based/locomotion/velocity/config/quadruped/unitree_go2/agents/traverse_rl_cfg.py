# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
RslRlOnPolicyRunnerCfg, 
RslRlPpoActorCriticCfg, 
RslRlPpoAlgorithmCfg,
)
#########################
# Policy configurations #
#########################

@configclass
class TraverseRslRlBaseCfg:
    num_priv_explicit: int = 3 + 3 + 3  # 9
    num_priv_latent: int = 4 + 1 + 12 + 12  # 29
    num_prop: int = 3 + 2 + 3 + 4 + 36 + 5  # 53
    num_scan: int = 132
    num_hist: int = 10
    
@configclass
class TraverseRslRlStateHistEncoderCfg(TraverseRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder" 
    channel_size: int = 10 
    
@configclass
class TraverseRslRlDepthEncoderCfg(TraverseRslRlBaseCfg):
    backbone_class_name: str = "DepthOnlyFCBackbone58x87" 
    encoder_class_name: str = "RecurrentDepthBackbone" 
    depth_shape: tuple[int] = (87, 58)
    hidden_dims: int = 512
    learning_rate: float = 1.e-3
    num_steps_per_env: int = 24 * 5

@configclass
class TraverseRslRlEstimatorCfg(TraverseRslRlBaseCfg):
    class_name: str = "DefaultEstimator" 
    train_with_estimated_states: bool = True 
    learning_rate: float = 1.e-4 
    hidden_dims: list[int] = MISSING 
    
@configclass
class TraverseRslRlActorCfg(TraverseRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: TraverseRslRlStateHistEncoderCfg = MISSING


@configclass
class TraverseRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = 'ActorCriticRMA'
    tanh_encoder_output: bool = False 
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    actor: TraverseRslRlActorCfg = MISSING

@configclass
class TraverseRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'PPOWithExtractor'
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float]= [0, 0.1, 2000, 3000]

@configclass
class TraverseRslRlDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DistillationWithExtractor"

@configclass
class TraverseRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: TraverseRslRlPpoActorCriticCfg = MISSING
    estimator: TraverseRslRlEstimatorCfg = MISSING
    depth_encoder: TraverseRslRlDepthEncoderCfg | None = None
    algorithm: TraverseRslRlPpoAlgorithmCfg | TraverseRslRlDistillationAlgorithmCfg = MISSING

