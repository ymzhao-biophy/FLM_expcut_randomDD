import numpy as np
import matplotlib.pyplot as plt
import parameter as para
import files
import sys
def get_msd(trajectories):
    N=len(trajectories[0])
    msd = np.zeros(N)
    for i in range(N):
        square_disp = trajectories[:, i] ** 2
        msd[i] = np.mean(square_disp)
    return msd


def plot_multiple_msd(distribution,process,decay_type, T,dt,H_list,N_traj,nu,mu,alpha,D_beta,gamma):
    msd_results = {}  # 存储每个 H 对应的 MSD
    common_ts = None  # 存储时间轴（所有 H 应该一样）
    for H in H_list:
        print(f"\n===== Running for H = {H} =====")
        fname_current=files.build_traj_ensemble_filename(distribution,process,decay_type,T,dt,H,N_traj,nu,mu,alpha,D_beta,gamma)
        ts, trajectories = files.read_trajectories_data(fname_current)
        if common_ts is None:
            common_ts = ts
        msd = get_msd(trajectories)
        msd_results[H] = msd
    plt.figure(figsize=(9, 6))
    for H, msd in msd_results.items():
        tau = common_ts[1:]  # τ > 0
        msd_plot = msd[1:]
        n_total = len(tau)
        if n_total > 10000:
            indices = np.linspace(0, n_total - 1, 1000, dtype=int)
            tau_sampled = tau[indices]
            msd_sampled = msd_plot[indices]
        else:
            tau_sampled = tau
            msd_sampled = msd_plot

        # 从采样点中取后30%进行拟合
        n_sampled = len(tau_sampled)
        start_fit = int(n_sampled * 0.3)
        tau_fit1 = tau_sampled[start_fit:]
        msd_fit1 = msd_sampled[start_fit:]
        if len(tau_fit1) >= 2:
            log_t1, log_m1 = np.log(tau_fit1), np.log(msd_fit1)
            slope1, intercept1 = np.polyfit(log_t1, log_m1, 1)
        t_line1 = np.logspace(np.log10(tau_fit1[0]), np.log10(tau_fit1[-1]), 100)
        msd_line1 = np.exp(intercept1) * t_line1 ** slope1
        plt.plot(t_line1, msd_line1, '-', color='black',linewidth=1.5,
                 label=f'fit 30% tail, slope={slope1:.3f} for H={H}')
        ##取中间一段进行拟合
        t_min, t_max = 1.0, 1.0 / nu
        mask2 = (tau_sampled >= t_min) & (tau_sampled <= t_max)
        tau_fit2 = tau_sampled[mask2]
        msd_fit2 = msd_sampled[mask2]
        log_t2, log_m2 = np.log(tau_fit2), np.log(msd_fit2)
        slope2, intercept2 = np.polyfit(log_t2, log_m2, 1)
        t_line2 = np.logspace(np.log10(t_min), np.log10(t_max), 100)
        msd_line2 = np.exp(intercept2) * t_line2 ** slope2
        plt.plot(t_line2, msd_line2, '--', color='black',linewidth=1.5,
                 label=f'fit [{t_min:.1f}, {t_max:.1f}], slope={slope2:.3f} for H={H}')

        plt.loglog()
        plt.scatter(tau_sampled, msd_sampled, label=f'H = {H}', s=5, alpha=0.6)

    plt.xlabel('time')
    plt.ylabel('MSD')
    if decay_type=='no':
        plt.title(f'{process} {decay_type}_MSD for different H')
    elif decay_type=='power':
        plt.title(f'{process} {decay_type} MSD for different H with tau={1 / nu} and mu={mu}')
    elif decay_type=='exponential':
        plt.title(f'{process} {decay_type} MSD for different H with tau={1 / nu}')
    plt.rcParams.update({
        'font.size': 12,  # 全局字体大小
        'axes.labelsize': 16,  # 坐标轴标签字体大小
        'axes.titlesize': 16,  # 标题字体大小
        'xtick.labelsize': 16,  # x轴刻度标签大小
        'ytick.labelsize': 16,  # y轴刻度标签大小
        'legend.fontsize': 12  # 图例字体大小
    })
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    distribution,process,decay_type= sys.argv[1],sys.argv[2],sys.argv[3]
    plot_multiple_msd(distribution,process,decay_type,para.T,para.dt,para.H_list,para.N_traj,para.nu,para.mu,para.alpha,para.D_beta,para.gamma)
