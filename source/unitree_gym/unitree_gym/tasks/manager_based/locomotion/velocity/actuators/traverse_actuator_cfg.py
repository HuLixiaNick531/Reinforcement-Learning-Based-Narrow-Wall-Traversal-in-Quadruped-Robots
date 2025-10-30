# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable

from isaaclab.utils import configclass
from isaaclab.actuators.actuator_cfg import DCMotorCfg
from . import traverse_actuator_pd

@configclass
class TraverseDCMotorCfg(DCMotorCfg):

    class_type: type = traverse_actuator_pd.TraverseDCMotor

    saturation_effort: dict[str, float] | None = None
    """Peak motor force/torque of the electric DC motor (in N-m)."""
