from isaaclab.utils import configclass
from ..traverse_terrain_generator_cfg import TraverseSubTerrainBaseCfg
from . import extreme_traverse_terrians

@configclass
class ExtremeTraverseRoughTerrainCfg(TraverseSubTerrainBaseCfg):
    apply_roughness: bool = True 
    apply_flat: bool = False 
    downsampled_scale: float | None = 0.075
    noise_range: tuple[float,float] = (0.02, 0.06)
    noise_step: float = 0.005
    x_range: tuple[float, float] = (0.8, 1.5)
    y_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width: tuple[float, float] = (0.6, 1.2)
    pad_width: float = 0.1 
    pad_height: float = 0.0

@configclass
class ExtremeTraverseTwoWallsTerrainCfg(ExtremeTraverseRoughTerrainCfg):
    function = extreme_traverse_terrians.two_walls_terrain   # ← 指向上面新增的函数
    # 两墙参数（米）
    # 关键参数（单位：米）
    platform_height = 0.0
    wall_height     = 0.35
    wall_thickness  = 0.20
    corridor_width  = 0.90     # 走廊的净宽（两墙之间）
    corridor_start_x= 2.0      # 入口距离机器人初始点的 x
    corridor_length = 3.0      # 走廊长度
    num_goals       = 3        # 入口前/中点/出口后
    apply_roughness = False    # 想要完全平整就关掉