import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from tqdm import tqdm
import time
import parameter as para
import files
# ========== Parameters ==========
nu = para.nu
H_list = para.H_list
T = para.T
dt = para.dt
N_traj = para.N_traj
mu = para.mu
alpha = para.alpha
D_beta = para.D_beta
gamma=para.gamma
lamda=para.lamda
K_val = 1.0
N_sample = 20
colors = ["#2196F3", "#FF5722"]

# ========== Empirical MSD ==========
def get_msd(trajectories):
    N = len(trajectories[0])
    msd = np.zeros(N)
    for i in range(N):
        square_disp = trajectories[:, i] ** 2
        msd[i] = np.mean(square_disp)
    return msd

# ========== Theoretical MSD (exponential decay) ==========
def exponential_flm_msd_single(t, H, nu, gamma, lamda, alpha):
    N = gamma*t + 2.0 * H
    gamma_ratio = mp.exp(mp.loggamma(N) - mp.loggamma(gamma*t))
    hyp1 = mp.hyp2f1(N, 2.0 * H - 1.0, 2.0 * H, -nu/lamda )
    hyp2 = mp.hyp2f1(N, 2.0 * H, 2.0 * H + 1.0, -nu/lamda)
    term1 = hyp1 / (2.0 * H - 1.0)
    term2 = hyp2 / (2.0 * H)
    bracket = term1 - term2
    other=2 * alpha * H * (2 * H - 1) * (1/lamda)**(2*H)
    return float(gamma_ratio * bracket * other)

def exponential_flm_msd(t_vals, H, nu, gamma,lamda, alpha, desc=None):
    t_vals = np.asarray(t_vals, dtype=float)
    results = np.zeros(len(t_vals))
    iterator = range(len(t_vals))
    if desc is not None:
        iterator = tqdm(iterator, total=len(t_vals), desc=desc, leave=False)
    for i in iterator:
        results[i] = exponential_flm_msd_single(t_vals[i], H, nu, gamma, lamda, alpha)
    return results

if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 22, "axes.labelsize": 22, "axes.titlesize": 22,
        "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
    })

    plt.figure(figsize=(11, 7))
    process = "flm"
    decay_type = "exponential"
    distribution= 'gamma'
    t_total_start = time.time()

    for idx, H in enumerate(tqdm(H_list, desc="Processing H values")):
        print()
        print(f"===== H = {H} =====")
        fname_current = files.build_traj_ensemble_filename(distribution, process, decay_type, T, dt, H, N_traj, nu, mu,
                                                           alpha, D_beta, gamma)
        ts, trajectories = files.read_trajectories_data(fname_current)
        print(f"  Trajectories shape: {trajectories.shape}, N_steps={len(trajectories[0])}")

        N_total = len(trajectories[0])

        # --- Empirical MSD ---
        print("Computing empirical MSD...")
        msd_emp = get_msd(trajectories)

        # --- Select N_sample log-spaced time points (skip t=0) ---
        t_min = ts[1]
        t_max = ts[-1]
        t_sample = np.logspace(np.log10(t_min), np.log10(t_max), N_sample)
        idx_sample = np.searchsorted(ts, t_sample)
        idx_sample = np.clip(idx_sample, 1, N_total - 1)
        idx_sample = np.unique(idx_sample)

        tau_sample = ts[idx_sample]
        msd_sample = msd_emp[idx_sample]
        print(f"  Sampled times: {np.round(tau_sample, 1).tolist()}")
        print(f"  Sampled MSD: {np.round(msd_sample, 2).tolist()}")

        # --- Theoretical MSD: dense for curve, sampled for circles ---
        t_dense = np.logspace(np.log10(t_min), np.log10(t_max), 100)
        print(f"Computing theoretical MSD (dense, {len(t_dense)} pts)...")
        msd_theory_dense = exponential_flm_msd(t_dense, H, nu, gamma, lamda, alpha, desc=f"Theory dense H={H}")

        # --- Plot ---
        color = colors[idx]
        plt.loglog(t_dense, msd_theory_dense, "-", color=color, linewidth=2,
                   label=f"H={H} (theory)")
        plt.loglog(tau_sample, msd_sample, "o",
                   markerfacecolor="none", markeredgecolor=color,
                   markersize=9, markeredgewidth=1.8,
                   label=f"H={H} (simulation)")
    # --- Reference: MSD = t (normal diffusion) in last 30% of time axis ---
    t_ref_start = t_dense[int(0.8 * len(t_dense))]
    t_ref = np.logspace(np.log10(t_ref_start), np.log10(t_dense[-1]), 50)
    plt.loglog(t_ref, t_ref, "k--", linewidth=2.5, alpha=0.7, label="MSD ∝ t")
    plt.xlabel("t")
    plt.ylabel("MSD")
    plt.legend(loc="upper left")
    #plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("msd_comparison.eps",format='eps', dpi=150)
    print()
    print("Saved: msd_comparison.eps")
    total_elapsed = time.time() - t_total_start
    print(f"Total time: {total_elapsed:.1f}s")
