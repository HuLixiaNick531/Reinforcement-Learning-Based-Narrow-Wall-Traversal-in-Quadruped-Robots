from ...traverse_terrain_generator_cfg import TraverseTerrainGeneratorCfg
from ...traverse_terrain_generator import TraverseTerrainGenerator
from ...extreme_traverse import *

TRAVERSE_TERRAINS_CFG = TraverseTerrainGeneratorCfg(
    class_type=TraverseTerrainGenerator,
    size=(16.0, 4.0),
    border_width=0.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08,  # original 0.05，但是这个值构建地面mesh失败，机器人直接穿过地面往下降
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),   # 用不到也保留字段
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "two_walls": ExtremeTraverseTwoWallsTerrainCfg(
            platform_height = 0.0,
            wall_height     = 0.5,
            wall_thickness  = 0.20,
            corridor_width  = 0.50,     # 走廊的净宽（两墙之间）
            corridor_start_x= 1.0,      # 入口距离机器人初始点的 x
            corridor_length = 3.0,      # 走廊长度
            num_goals       = 3,        # 入口前/中点/出口后
            apply_roughness = False,    # 想要完全平整就关掉
        ),
    },
)