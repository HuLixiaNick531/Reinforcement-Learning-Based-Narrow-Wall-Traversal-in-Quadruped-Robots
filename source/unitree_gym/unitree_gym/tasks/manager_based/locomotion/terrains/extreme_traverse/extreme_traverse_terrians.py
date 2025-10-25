
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
    cfg,          # 若担心循环依赖，可省类型注解
    num_goals: int,
):
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale

    # 分辨率（像素）
    width_px  = int(cfg.size[0] / hs)
    length_px = int(cfg.size[1] / hs)
    mid_y     = length_px // 2

    # 平台
    h_raw = np.zeros((width_px, length_px), dtype=np.int16)
    base_h_idx = int(round(cfg.platform_height / vs))
    h_raw[:, :] = base_h_idx

    # 参数（米 → 像素 / 索引）
    wall_h_idx   = int(round(cfg.wall_height / vs))
    wall_t_px    = max(1, int(round(cfg.wall_thickness / hs)))
    corridor_w_px= max(1, int(round(cfg.corridor_width / hs)))
    x0_px        = int(round(cfg.corridor_start_x / hs))
    x1_px        = min(width_px, x0_px + int(round(cfg.corridor_length / hs)))

    # 两面平行墙（沿 x 方向）
    half_gap = corridor_w_px // 2
    # 左墙在中线左侧： [mid_y - half_gap - wall_t, mid_y - half_gap)
    yL0 = max(0, mid_y - half_gap - wall_t_px)
    yL1 = max(0, mid_y - half_gap)
    # 右墙在中线右侧： [mid_y + half_gap, mid_y + half_gap + wall_t)
    yR0 = min(length_px, mid_y + half_gap)
    yR1 = min(length_px, mid_y + half_gap + wall_t_px)

    x0_px = max(0, x0_px)
    x1_px = max(x0_px + 1, x1_px)  # 至少 1 像素长度

    # 抬高到墙顶高度
    if yL0 < yL1:
        h_raw[x0_px:x1_px, yL0:yL1] = base_h_idx + wall_h_idx
    if yR0 < yR1:
        h_raw[x0_px:x1_px, yR0:yR1] = base_h_idx + wall_h_idx

    # 目标点（像素坐标）
    g0 = np.array([max(0, x0_px - int(round(0.6 / hs))), mid_y], dtype=np.int32)                # 入口前
    g1 = np.array([(x0_px + x1_px) // 2,                mid_y], dtype=np.int32)                 # 走廊中点
    g2 = np.array([min(width_px - 1, x1_px + int(round(0.6 / hs))), mid_y], dtype=np.int32)     # 出口后

    goals_px = np.stack([g0, g1, g2], axis=0)
    # 根据 num_goals 截断/补齐
    if num_goals <= goals_px.shape[0]:
        goals_px = goals_px[:num_goals]
    else:
        pad = np.repeat(goals_px[-1][None, :], num_goals - goals_px.shape[0], axis=0)
        goals_px = np.concatenate([goals_px, pad], axis=0)

    # 目标高度（按地面/平台高度；如需墙顶目标可自行改）
    goal_h_idx = h_raw[goals_px[:, 0], goals_px[:, 1]].astype(np.int16)

    # 可选：粗糙度；若想走廊完全平整，建议关闭 apply_roughness 或在这里跳过走廊区域
    if getattr(cfg, "apply_roughness", False):
        h_raw = random_uniform_terrain(difficulty, cfg, h_raw)

    # 返回：高度场（索引单位）、goals（米）、goal_heights（米）
    goals_m   = goals_px.astype(np.float32) * hs
    goal_h_m  = goal_h_idx.astype(np.float32) * vs
    return h_raw, goals_m, goal_h_m
