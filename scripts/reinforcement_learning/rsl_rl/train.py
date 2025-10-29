import numpy as np

# -----------------------------
# 1. 读取纯文本 NGSIM 文件
# -----------------------------
filename = 'ngsim_data.csv'  # 修改为你的文件名

# 加载数据，默认空格/制表符分隔
data = np.loadtxt(filename)

# -----------------------------
# 2. 筛选某辆车
# -----------------------------
vehicle_id = 437
mask = data[:,1] == vehicle_id
data_v = data[mask]

# 提取列
t = data_v[:,3] / 1e9        # 时间纳秒 -> 秒
x = data_v[:,4]              # x 坐标
y = data_v[:,5]              # y 坐标
heading_deg = data_v[:,11]   # 航向角度列（假设为第12列，索引11）
heading = np.radians(heading_deg)
speed_ngsim = data_v[:,13]   # 速度列（假设为第14列，索引13）

# -----------------------------
# 3. 初始化列表
# -----------------------------
v_b_list = []
v_s_list = []

# -----------------------------
# 4. 计算速度并验证
# -----------------------------
for i in range(len(data_v)-1):
    dt = t[i+1] - t[i]
    if dt == 0:
        continue  # 避免除零

    # 位置差分
    dp = np.array([x[i+1]-x[i], y[i+1]-y[i], 0.0]) / dt

    # body frame -> space frame 旋转矩阵
    R = np.array([
        [np.cos(heading[i]), -np.sin(heading[i]), 0],
        [np.sin(heading[i]),  np.cos(heading[i]), 0],
        [0, 0, 1]
    ])

    # 刚体速度 v_b
    v_b = R.T @ dp

    # 空间角速度 ω_s
    dtheta = heading[i+1] - heading[i]
    omega_s = np.array([0,0,dtheta/dt])

    # 空间速度 v_s
    p = np.array([x[i], y[i], 0.0])
    v_s = dp + np.cross(omega_s, -p)

    v_b_list.append(v_b)
    v_s_list.append(v_s)

    # 验证
    speed_error = abs(v_b[0] - speed_ngsim[i])
    vs_check = R @ v_b + np.cross(omega_s, p)
    formula_error = np.linalg.norm(v_s - vs_check)

    print(f"Frame {i}:")
    print(f"  v_b = {v_b}, v_s = {v_s}")
    print(f"  Speed error (v_b[0] vs NGSIM) = {speed_error:.3f} m/s")
    print(f"  Formula consistency error = {formula_error:.6f} m/s\n")

# -----------------------------
# 5. 统计最大误差
# -----------------------------
speed_errors = [abs(v[0]-s) for v,s in zip(v_b_list, speed_ngsim[:-1])]
formula_errors = [np.linalg.norm(vs - (np.array([[np.cos(h), -np.sin(h),0],[np.sin(h), np.cos(h),0],[0,0,1]]) @ vb + np.cross(np.array([0,0,(heading[i+1]-heading[i])/(t[i+1]-t[i])]), np.array([x[i],y[i],0])))) for i,(vb,vs,h) in enumerate(zip(v_b_list,v_s_list,heading[:-1]))]

print(f"Max speed error: {np.max(speed_errors):.3f} m/s")
print(f"Max formula consistency error: {np.max(formula_errors):.6f} m/s")
