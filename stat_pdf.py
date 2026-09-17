import numpy as np
import matplotlib.pyplot as plt
import parameter as para
import sys
import files
Target_time=30
Target_times=[0.5,5,20,100]
def plot_pdf(process,decay_type, plot_type, ts, trajectories, target_time, H, nu, mu):
    # 找到最接近目标时间的时间步索引
    idx = np.argmin(np.abs(ts - target_time))
    t = ts[idx]
    displacements = trajectories[:, idx]
    # 设置区间数
    bins = 50
    density, bin_edges = np.histogram(displacements, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if plot_type == 'log':
        nonzero_mask = density > 0
        x_plot = bin_centers[nonzero_mask]
        x_plot_rescale = x_plot
        lnp = np.log(density[nonzero_mask])
        plt.figure(figsize=(7, 5))
        plt.rcParams.update({
            'font.size': 18,  # 全局字体大小
            'axes.labelsize': 20,  # 坐标轴标签字体大小
            'axes.titlesize': 20,  # 标题字体大小
            'xtick.labelsize': 20,  # x轴刻度标签大小
            'ytick.labelsize': 20,  # y轴刻度标签大小
            'legend.fontsize': 18  # 图例字体大小
        })
        plt.xlim(-50, 50)
        plt.scatter(x_plot_rescale, lnp, color='blue', s=20, alpha=0.7,
                    label=f'Simulation (t≈{t:.3f}) for lnp-x with tau={1/nu}')
        plt.xlabel('x')
        plt.ylabel('log P')
        plt.title(f'PDF for{process} with {decay_type} decay at t = {t:.3f}, H = {H}')


    else:  # 默认 plot_type 为普通 PDF
        x_plot = bin_centers
        plt.figure(figsize=(7, 5))
        plt.rcParams.update({
            'font.size': 12,  # 全局字体大小
            'axes.labelsize': 16,  # 坐标轴标签字体大小
            'axes.titlesize': 16,  # 标题字体大小
            'xtick.labelsize': 16,  # x轴刻度标签大小
            'ytick.labelsize': 16,  # y轴刻度标签大小
            'legend.fontsize': 14  # 图例字体大小
        })
        plt.scatter(x_plot, density, color='blue', s=10, alpha=0.7,
                    label=f'Simulation (t≈{t:.3f})')
        plt.xlabel('x')
        plt.ylabel('P')
        plt.xlim(-50, 50)
        plt.title(f'PDF for {process} with {decay_type} decay at t = {t:.3f}, H = {H} with tau={1/nu}')


    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    distribution,process,decay_type,plot_type = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
    fname=files.build_traj_ensemble_filename(distribution,process,decay_type,para.T,para.dt,para.H,para.N_traj,para.nu,para.mu,para.alpha,para.D_beta,para.gamma)
    ts,trajectories=files.read_trajectories_data(fname)
    for Tar_time in Target_times:
        plot_pdf(process,decay_type, plot_type, ts, trajectories, Tar_time, para.H, para.nu, para.mu)
