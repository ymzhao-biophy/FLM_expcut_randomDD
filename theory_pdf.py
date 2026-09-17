import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gammainc, gamma as gamma_func, loggamma, kv
from scipy.integrate import quad
import parameter as para
import files
import os
import time
import tqdm
# ========== Parameters ==========
H_list  = para.H_list
nu      = para.nu
D_beta  = para.D_beta
alpha   = para.alpha          # Gamma distribution shape parameter for D
gamma   = para.gamma          # Gamma subordinator shape rate
lamda   = para.lamda          # Gamma subordinator scale (rate)
tau     = 1.0 / nu            # cutoff timescale
N_traj  = para.N_traj
Target_times = [0.5, 5, 20, 100]
BINS = 50
SQRT2 = np.sqrt(2.0)

_CORR = 2.0**(5.0 / 4.0)
_PHI0_PREFAC = _CORR * gamma_func(alpha - 0.5) / (np.sqrt(np.pi) * gamma_func(alpha) * 2.0**(alpha / 2.0 + 0.75))

def sigma2(s):
    a1 = 2.0 * H - 1.0
    a2 = 2.0 * H
    x = s / tau
    term = x * gamma_func(a1) * gammainc(a1, x) - gamma_func(a2) * gammainc(a2, x)
    return 2.0 * D_beta * H * (2.0 * H - 1.0) * tau**(2.0 * H) * term

def phi(x, s):
    s2 = sigma2(s)
    sg = np.sqrt(s2)
    if abs(x) < 1e-14:
        return _PHI0_PREFAC / sg
    z = SQRT2 * abs(x) / sg
    prefactor = _CORR * 2.0**(0.5 - alpha) / (np.sqrt(np.pi) * gamma_func(alpha))
    return prefactor * abs(x)**(alpha - 0.5) / sg**(alpha + 0.5) * kv(alpha - 0.5, z)

def rho(x, t):
    gt = gamma * t
    logC = gt * np.log(lamda) - loggamma(gt)
    s_mean = gt / lamda
    s_std  = np.sqrt(gt) / lamda
    lo = max(1e-12, s_mean - 10.0 * s_std)
    hi = s_mean + 10.0 * s_std + 10.0
    if gt < 2.0:
        lo = 0.0
    def integrand(s):
        log_weight = (gt - 1.0) * np.log(np.maximum(s, 1e-300)) - lamda * s + logC
        return phi(x, s) * np.exp(log_weight)
    res, _err = quad(integrand, lo, hi, limit=200, points=[1e-10] if lo == 0.0 else None)
    return res

def rho_array(x_vals, t):
    return np.array([rho(x, t) for x in x_vals])

if __name__ == '__main__':
    distribution = 'gamma'
    process      = 'flm'
    decay_type   = 'exponential'
    colors = ["#2196F3", "#FF5722"]
    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
    })

    for t_target in Target_times:
        plt.figure(figsize=(11, 7))
        for h_idx, H in enumerate(tqdm.tqdm(H_list, desc="Processing H values")):
            color = colors[h_idx]
            fname = files.build_traj_ensemble_filename(distribution,process,decay_type,para.T,para.dt,H,N_traj,nu,para.mu,alpha,D_beta,gamma)
            path = os.path.join(files.save_dir, fname)
            print(f'Memory-mapping: {fname}')
            t0 = time.time()
            data = np.load(path, mmap_mode='r')
            ts = data['ts']
            trajectories = data['trajectories']
            print(f'  loaded in {time.time()-t0:.1f}s, 'f'shape: {trajectories.shape}')
            time_idx = np.argmin(np.abs(ts - t_target))
            t_actual = ts[time_idx]
            print(
                f'Reading displacements for '
                f't={t_target:.1f} '
                f'(idx={time_idx}, actual t={t_actual:.3f}) ...')
            displacements = trajectories[:, time_idx]

            # Simulation histogram
            density, bin_edges = np.histogram(displacements,bins=BINS,density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mask = density > 1e-4
            x_sim = bin_centers[mask]
            y_sim = density[mask]

            # Theory
            t1 = time.time()
            y_theory = rho_array(bin_centers,t_target)
            print(f'Theory calculated in 'f'{time.time()-t1:.1f}s')

            # Simulation
            plt.plot(x_sim,y_sim,'o',markerfacecolor="none",markeredgecolor=color,markersize=9,markeredgewidth=1.8,linestyle='none',label=f'Simulation, $H={H}$')

            # Theory
            plt.plot(bin_centers,y_theory,'-',color=color,linewidth=2,label=f'Theory, $H={H}$')
        # ===== Figure settings =====

        plt.yscale('log')
        plt.ylim(1e-4, None)
        plt.xlim(-50, 50)
        plt.xlabel('x')
        plt.ylabel(r'$P(x,t)$')
        plt.title(f't = {t_target}')
        #plt.legend()
        plt.tight_layout()
        outpath = f'pdf_T={t_target}.eps'
        plt.savefig(outpath,format='eps',dpi=150,bbox_inches='tight')
        plt.close()
        print(f'Saved to {outpath}')
