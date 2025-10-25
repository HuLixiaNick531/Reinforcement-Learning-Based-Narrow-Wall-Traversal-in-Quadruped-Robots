# walls.py
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for a flat tile with two parallel walls using Mesh terrains."""

from __future__ import annotations
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# ========== 可调参数（单位: m） ==========
TERRAIN_SIZE_X = 8.0
TERRAIN_SIZE_Y = 8.0

WALL_LENGTH     = 6.0    # 墙沿 X 的长度
WALL_THICKNESS  = 0.20   # 墙沿 Y 的厚度
WALL_HEIGHT     = 0.60   # 墙的高度（Z）
WALL_SEPARATION = 1.20   # 两道墙中心距（沿 Y）

# 相对 tile 中心的小偏移（如需把墙整体平移）
WALL_CENTER_X = 0.0
WALL_CENTER_Y = 0.0

# 网格分辨率（多数用于高度场；这里保留与其它地形统一风格）
HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE   = 0.005


def _build_mesh_rectangles_cfg():
    """优先：使用矩形网格体在平地上放两道墙。"""
    # rectangles: (center_x, center_y, size_x, size_y, height)
    rectangles = [
        (WALL_CENTER_X, WALL_CENTER_Y + WALL_SEPARATION / 2.0,
         WALL_LENGTH,   WALL_THICKNESS, WALL_HEIGHT),
        (WALL_CENTER_X, WALL_CENTER_Y - WALL_SEPARATION / 2.0,
         WALL_LENGTH,   WALL_THICKNESS, WALL_HEIGHT),
    ]
    return terrain_gen.MeshRectanglesTerrainCfg(
        proportion=1.0,
        rectangles=rectangles,
        # 下列字段名可能随版本略变，如报签名错误请删改
        platform_width=max(2.0, WALL_LENGTH + 0.5),
        border_width=0.25,
        holes=False,
    )


def _build_mesh_boxes_cfg():
    """回退：如果没有 MeshRectanglesTerrainCfg，则用 '盒子' 方式放两道墙。"""
    # boxes: (center_x, center_y, size_x, size_y, height)
    boxes = [
        (WALL_CENTER_X, WALL_CENTER_Y + WALL_SEPARATION / 2.0,
         WALL_LENGTH,   WALL_THICKNESS, WALL_HEIGHT),
        (WALL_CENTER_X, WALL_CENTER_Y - WALL_SEPARATION / 2.0,
         WALL_LENGTH,   WALL_THICKNESS, WALL_HEIGHT),
    ]
    return terrain_gen.MeshBoxesTerrainCfg(
        proportion=1.0,
        boxes=boxes,
        border_width=0.25,
        # 某些版本可能支持 platform_width/holes 等参数，如报错可去掉
    )


# -------------- 导出总配置 --------------
try:
    _sub_terrain_cfg = _build_mesh_rectangles_cfg()
except Exception:
    _sub_terrain_cfg = _build_mesh_boxes_cfg()

WALL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(TERRAIN_SIZE_X, TERRAIN_SIZE_Y),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=HORIZONTAL_SCALE,
    vertical_scale=VERTICAL_SCALE,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat_with_two_walls": _sub_terrain_cfg,
    },
)
"""Two parallel walls on a flat field."""