
from __future__ import annotations

import numpy as np
import random
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING
from ..utils import traverse_field_to_mesh
if TYPE_CHECKING:
    from .config import extreme_traverse_terrains_cfg

"""
Reference from https://arxiv.org/pdf/2309.14341
"""

def padding_height_field_raw(
    height_field_raw:np.ndarray, 
    cfg:extreme_traverse_terrains_cfg.ExtremeTraverseRoughTerrainCfg
    )->np.ndarray:
    pad_width = int(cfg.pad_width // cfg.horizontal_scale)
    pad_height = int(cfg.pad_height // cfg.vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height
    height_field_raw = np.rint(height_field_raw).astype(np.int16)
    return height_field_raw

def random_uniform_terrain(
    difficulty: float, 
    cfg: extreme_traverse_terrains_cfg.ExtremeTraverseRoughTerrainCfg,
    height_field_raw: np.ndarray,
    ):
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    max_height = (cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]
    height_min = int(-cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(max_height / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    z_upsampled = np.rint(z_upsampled).astype(np.int16)
    height_field_raw += z_upsampled 
    return height_field_raw 


@traverse_field_to_mesh
def two_walls_terrain(
    difficulty: float, 
    cfg, 
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单段双墙走廊地形"""

    # ------------------------ 基本分辨率设置 ------------------------
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale
    width_pixels = int(cfg.size[0] / hs)
    length_pixels = int(cfg.size[1] / hs)
    mid_y = length_pixels // 2

    # ------------------------ 初始化地形 ------------------------
    height_field_raw = np.zeros((width_pixels, length_pixels))
    platform_len = round(cfg.corridor_start_x / hs)
    platform_height = round(cfg.platform_height / vs)
    height_field_raw[0:platform_len, :] = platform_height

    # ------------------------ 参数设置 ------------------------
    wall_t_px = max(1, int(round(cfg.wall_thickness / hs)))
    base_gap_px = int(round(cfg.corridor_width / hs))
    wall_h_idx = round(cfg.wall_height / vs)
    seg_len = round(cfg.corridor_length / hs)

    # ------------------------ 构造单段走廊 ------------------------
    x0 = platform_len
    x1 = x0 + seg_len
    half_gap = base_gap_px // 2

    # 左右墙的像素范围
    yL0 = max(0, mid_y - half_gap - wall_t_px)
    yL1 = max(0, mid_y - half_gap)
    yR0 = min(length_pixels, mid_y + half_gap)
    yR1 = min(length_pixels, mid_y + half_gap + wall_t_px)

    # 填充墙高
    if yL0 < yL1:
        height_field_raw[x0:x1, yL0:yL1] = platform_height + wall_h_idx
    if yR0 < yR1:
        height_field_raw[x0:x1, yR0:yR1] = platform_height + wall_h_idx

    # ------------------------ 目标点 ------------------------
    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height

    # goal[0]: 入口前
    goals[0] = [platform_len - 1, mid_y]
    # goal[1]: 走廊中点
    goals[1] = [x0 + seg_len // 2, mid_y]
    # goal[2]: 走廊出口后
    goals[-1] = [min(width_pixels - 1, x1 + int(0.6 / hs)), mid_y]

    # ------------------------ 边缘填充 ------------------------
    height_field_raw = padding_height_field_raw(height_field_raw, cfg)

    # ------------------------ 可选粗糙度 ------------------------
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    # ------------------------ 输出转换（像素→米） ------------------------
    return (
        height_field_raw,
        goals * cfg.horizontal_scale,
        goal_heights * cfg.vertical_scale,
    )