import numpy as np
import matplotlib.pyplot as plt
import parameter as para
import files
import sys
def get_kurtosis(trajectories,time_cut):
    N=len(trajectories[0])
    msd = np.zeros(N)
    mfd = np.zeros(N)# x^4
    kurtosis=np.zeros(N)
    for i in range(time_cut,N):
        square_disp = trajectories[:, i] ** 2
        fourth_disp = trajectories[:, i] ** 4
        msd[i] = np.mean(square_disp)
        mfd[i] = np.mean(fourth_disp)
        kurtosis[i]=mfd[i]/msd[i]**2
    return kurtosis

def plot_multiple_kurtosis(distribution,process,decay_type, T,dt,H_list,N_traj,nu,mu,alpha,D_beta,gamma,time_cut=1):
    kurtosis_result = {}
    common_ts = None  # 存储时间轴（所有 H 应该一样）
    for H in H_list:
        print(f"\n===== Running for H = {H} =====")
        fname_current = files.build_traj_ensemble_filename(distribution,process,decay_type, T, dt, H, N_traj, nu, mu, alpha, D_beta,gamma)
        ts, trajectories = files.read_trajectories_data(fname_current)
        if common_ts is None:
            common_ts = ts
        kurtosis = get_kurtosis(trajectories,time_cut)
        kurtosis_result[H] = kurtosis
    plt.figure(figsize=(9, 6))
    for H, kurtosis in kurtosis_result.items():
        tau = common_ts[time_cut:]
        k_plot = kurtosis[time_cut:]
        plt.loglog(tau, k_plot, label=f'H ={H}', linewidth=2)
    plt.axhline(y=4.5, color='red', linestyle='--', label='y=3')
    plt.xlabel('time')
    plt.ylabel('Kurtosis')
    if decay_type=='no':
        plt.title(f'{distribution}D_{process}_{decay_type}_kurtosis for different H')
    elif decay_type=='power':
        plt.title(f'{distribution}D_{process}_{decay_type} kurtosis for different H with tau={1 / nu} and mu={mu}')
    elif decay_type=='exponential':
        plt.title(f'{distribution}D_{process}_{decay_type} kurtosis for different H with tau={1 / nu}')
    plt.rcParams.update({
        'font.size': 12,  # 全局字体大小
        'axes.labelsize': 20,  # 坐标轴标签字体大小
        'axes.titlesize': 16,  # 标题字体大小
        'xtick.labelsize': 16,  # x轴刻度标签大小
        'ytick.labelsize': 16,  # y轴刻度标签大小
        'legend.fontsize': 12  # 图例字体大小
    })
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig("kurtosis.eps",format='eps', dpi=150)

if __name__ == "__main__":
    distribution,process,decay_type= sys.argv[1],sys.argv[2],sys.argv[3]
    plot_multiple_kurtosis(distribution,process,decay_type,para.T,para.dt,para.H_list,para.N_traj,para.nu,para.mu,para.alpha,para.D_beta,para.gamma)
