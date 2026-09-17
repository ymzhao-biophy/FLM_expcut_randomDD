import os
##write data into file
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)   # 若目录不存在则自动创建
# ========== 基础参数 ==========
T       = 1000        # 总时间
dt      = 0.1    # 物理时间步长
nu      = 0.1    # 衰减特征时间倒数
mu      = 0.6      #关联函数的幂率衰减
gamma   = 10     # Gamma 过程形状参数
lamda   = gamma        # Gamma 过程尺度参数（注意与 Python 关键字 lambda 区分）
h       = dt / 2    # FBM 内部时间步长
H       = 0.7      # Hurst 指数
alpha   = 2
D_beta  = 1
D_ensemble_num = 5000
N_eachD = 40
N_traj = D_ensemble_num * N_eachD # 统计所用轨迹数
H_list = [0.7,0.9]
# ========== 派生量 ==========
traj_lenth       = int(T / dt)                # 物理步数
Target_time=1
