import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import files
import parameter as para
# ========== Parameters ==========
nu = para.nu
H_list = para.H_list
T = para.T
dt = para.dt
N_traj = para.N_traj
mu = para.mu
alpha = para.alpha
D_beta = para.D_beta
N_sample = 20 ##numbers of points of lag time
gamma=para.gamma
colors = ["#2196F3", "#FF5722"]



def single_traj_moment(ts, trajectory, lag_time_index):
    """Compute MSD and 4th moment for a single trajectory at given lag time."""
    k =lag_time_index
    displacement = trajectory[k:] - trajectory[:-k]
    single_msd = np.mean(displacement ** 2)
    single_fmoment = np.mean(displacement ** 4)
    return single_msd, single_fmoment


def non_gaussian_parameter(ts, trajectories, lag_time_index):
    """Non-Gaussian parameter: <x^4>/(3<x^2>^2) - 1, averaged over ensemble."""
    N_t = trajectories.shape[0]
    msd_ensemble = np.zeros(N_t)
    fmoment_ensemble = np.zeros(N_t)
    for i in range(N_t):
        trajectory = trajectories[i, :]
        msd_ensemble[i], fmoment_ensemble[i] = single_traj_moment(ts, trajectory, lag_time_index)
    ngp = (np.mean(fmoment_ensemble) / np.mean(msd_ensemble) ** 2) / 3 - 1
    return ngp


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 22, "axes.labelsize": 22, "axes.titlesize": 22,
        "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
    })

    plt.figure(figsize=(11, 7))

    process = "flm"
    decay_type = "exponential"
    distribution = "gamma"
    t_total_start = time.time()

    for idx, H in enumerate(tqdm(H_list, desc="Processing H values")):
        print()
        print(f"===== H = {H} =====")
        fname_current = files.build_traj_ensemble_filename(
            distribution, process, decay_type, T, dt, H, N_traj, nu, mu, alpha, D_beta,gamma)
        ts, trajectories = files.read_trajectories_data(fname_current)

        # --- Lag times: log-spaced, up to 5% of total time ---
        t_min = ts[1]
        t_max = ts[-1] * 0.1
        lag_times = np.logspace(np.log10(t_min), np.log10(t_max), N_sample)
        lagtimes_raw_index=lag_times/dt
        lag_times_index=lagtimes_raw_index.astype(int)
        lag_times_index = np.maximum(lag_times_index, 1)
        lag_times_index=np.unique(lag_times_index)
        lag_times_new=lag_times_index*dt
        print(lag_times_index)

        ngp_vals = np.zeros(len(lag_times_index))
        for i, tau_index in enumerate(tqdm(lag_times_index, desc=f"NGP H={H}", leave=False)):
            ngp_vals[i] = non_gaussian_parameter(ts, trajectories, tau_index)

        color = colors[idx]
        plt.plot(lag_times_new, ngp_vals, "o",
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=9, markeredgewidth=1.8,
                 label=rf"$H={H}(simulation)$")
        print(f'for H={H},non-gaussian_parameter={ngp_vals}')

    plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.2, alpha=0.7, label=rf"$\Omega(\tau)=1/\alpha$")
    plt.xscale('log')    # x-axis in log scale
    plt.yscale('log')
    plt.ylim(0.1,10)
    plt.xlabel(r"Lag time $\tau$")
    plt.ylabel(r"Non-Gaussian parameter $\Omega(\tau)$")
    plt.legend(loc="upper right")
    #plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("non_gaussian_parameter.eps",format='eps', dpi=150)
    print()
    print("Saved: non_gaussian_parameter.eps")

    total_elapsed = time.time() - t_total_start
    print(f"Total time: {total_elapsed:.1f}s")
