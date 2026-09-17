import numpy as np
import parameter as para
import files
import matplotlib.pyplot as plt
from scipy.special import gammaln, roots_legendre
from tqdm import tqdm
import time

# ========== Parameters ==========
nu = para.nu
alpha = para.alpha          # Gamma distribution shape parameter for D
T = para.T
dt = para.dt
mu = para.mu
D_beta = para.D_beta
gamma = para.gamma
lamda = para.lamda
H_list = para.H_list
N_traj = para.N_traj
N_sample = 20
time_cut = 1               # skip first 10 points
distribution = 'gamma'
process = 'flm'
decay_type = 'exponential'
colors = ['#2196F3', '#FF5722']

# ========== Gauss-Legendre quadrature ==========
N_GL = 128
gl_nodes, gl_weights = roots_legendre(N_GL)
u_gl = 0.5 * (gl_nodes + 1.0)
w_gl = 0.5 * gl_weights
U2d, V2d = np.meshgrid(u_gl, u_gl, indexing='ij')
W2d = np.outer(w_gl, w_gl)

def gl_quad_1d(f_vals):
    return np.dot(w_gl, f_vals)

def gl_quad_2d(f_vals_2d):
    return np.sum(W2d * f_vals_2d)

def compute_kurtosis_theory(t_vals, H, nu, gamma, lamda, desc=None):
    nu_per_lamda = nu / lamda
    alpha_tr = 1.0 / (2.0 * H - 1.0)
    ua = u_gl ** alpha_tr
    ua_m1 = ua - 1.0
    nu_ua = nu_per_lamda * ua
    ua_2d = U2d ** alpha_tr
    va_2d = V2d ** alpha_tr
    ua2d_m1 = ua_2d - 1.0
    va2d_m1 = va_2d - 1.0

    K_vals = np.zeros(len(t_vals))
    iterator = enumerate(t_vals)
    if desc is not None:
        iterator = tqdm(iterator, total=len(t_vals), desc=desc, leave=False)
    for i, t in iterator:
        gt = gamma * t

        # B(t): eq (19)
        denom_B = (1.0 + nu_ua) ** (gt + 2.0 * H)
        integrand_B = alpha_tr * ua_m1 / denom_B
        B_sqrt = gl_quad_1d(integrand_B)
        B = B_sqrt * B_sqrt

        # A(t): eq (18)
        denom_A = (1.0 + nu_per_lamda * (ua_2d + va_2d)) ** (gt + 4.0 * H)
        integrand_A = alpha_tr * alpha_tr * ua2d_m1 * va2d_m1 / denom_A
        A = gl_quad_2d(integrand_A)

        log_ratio = gammaln(gt + 4.0 * H) + gammaln(gt) - 2.0 * gammaln(gt + 2.0 * H)
        K_vals[i] = 3.0 * np.exp(log_ratio) * A / B
    return K_vals

def get_kurtosis(trajectories, time_cut):
    N = len(trajectories[0])
    kurtosis = np.zeros(N)
    for i in range(time_cut, N):
        square_disp = trajectories[:, i] ** 2
        fourth_disp = trajectories[:, i] ** 4
        msd = np.mean(square_disp)
        mfd = np.mean(fourth_disp)
        kurtosis[i] = mfd / (msd * msd)
    return kurtosis

if __name__ == '__main__':
    # Gamma-D factor from eq (17)
    if distribution=='gamma':
        factor = 1.0 + 1.0 / alpha
    else:
        factor= 1.0

    plt.figure(figsize=(11, 7))
    plt.rcParams.update({
        'font.size': 22, 'axes.labelsize': 22, 'axes.titlesize': 22,
        'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18,
    })

    for idx, H in enumerate(tqdm(H_list, desc='Processing H values')):
        print()
        print(f'===== H = {H} =====')

        fname_current = files.build_traj_ensemble_filename(
            distribution, process, decay_type, T, dt, H, N_traj, nu, mu, alpha, D_beta, gamma)
        ts, trajectories = files.read_trajectories_data(fname_current)
        N_total = len(trajectories[0])

        print('Computing empirical kurtosis...')
        k_emp = get_kurtosis(trajectories, time_cut)

        t_min = ts[time_cut]
        t_max = ts[-1]
        t_sample = np.logspace(np.log10(t_min), np.log10(t_max), N_sample)
        idx_sample = np.searchsorted(ts, t_sample)
        idx_sample = np.clip(idx_sample, time_cut, N_total - 1)
        idx_sample = np.unique(idx_sample)

        tau_sample = ts[idx_sample]
        k_emp_sample = k_emp[idx_sample]

        t_dense = np.logspace(np.log10(t_min), np.log10(t_max), 200)
        print(f'Computing theoretical kurtosis (dense, {len(t_dense)} pts)...')
        k_theory_dense = compute_kurtosis_theory(t_dense, H, nu, gamma, lamda, desc=f'Theory dense H={H}')
        k_theory_dense *= factor

        color = colors[idx]
        plt.plot(t_dense, k_theory_dense, '-', color=color, linewidth=2,
                   label=rf'H={H} (theory)')
        plt.plot(tau_sample, k_emp_sample, 'o',
                   markerfacecolor='none', markeredgecolor=color,
                   markersize=9, markeredgewidth=1.8,
                   label=f'H={H} (simulation)')

    plt.axhline(y=3.0 * factor, color='k', linestyle='--', linewidth=1.2, alpha=0.7,
                label=rf'K = 3(1+1/$\alpha$) ')
    plt.xscale('log')    # x-axis in log scale
    plt.yscale('log')
    plt.ylim(1,100)
    plt.xlabel('Time t')
    plt.ylabel(r"Kurtosis $\kappa(t)$")
    plt.legend(loc='upper right')
    #plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('kurtosis_comparison_gammaD.eps',format='eps', dpi=150)
    print()
    print('Saved: kurtosis_comparison_gammaD.eps')
