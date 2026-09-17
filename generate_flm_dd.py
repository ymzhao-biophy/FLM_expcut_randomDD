import numpy as np
import FBM_trajectory as FBM
import FLM_trajectory as FLM
from tqdm import tqdm
import files
import parameter as para
import sys

N_traj=para.N_traj
T = para.T
dt = para.dt
nu = para.nu
mu = para.mu
gamma = para.gamma
lamda = para.lamda
h = para.h
H = para.H
traj_lenth=para.traj_lenth
H_list = para.H_list
alpha= para.alpha
D_beta=para.D_beta
D_ensemble_num=para.D_ensemble_num
N_eachD=para.N_eachD
mean_S = gamma * T / lamda
std_S = np.sqrt(gamma * T) / lamda
tau_max = mean_S + 6 * std_S
N_fbm_fixed = int(np.ceil(tau_max / h))

def get_fbm_ensemble(distribution,process,decay_type,N_eachD,traj_lenth,dt,H,nu,mu,D_ensemble,alpha,D_beta,gamma,save=True):
    m, lamda = FBM.get_circulant_eigenvalue(decay_type, traj_lenth, dt, H,nu,mu)
    D_numbers=len(D_ensemble)
    N_traj=D_numbers*N_eachD
    trajectories = np.zeros((N_traj, traj_lenth + 1))
    for i in tqdm(range(D_numbers), desc=f"fbm trajectory for H={H}"):
        for j in range(N_eachD):
            trajectories[i*N_eachD+j] = FBM.generate_FBM_trajectory(m, D_ensemble[i] * lamda, traj_lenth)
    T = traj_lenth * dt
    ts = np.linspace(0, T, traj_lenth + 1)
    if save:
        files.save_trajectories_to_file(distribution,process,decay_type,T, dt, H, N_traj, nu, mu, alpha, D_beta, ts, trajectories,gamma)
    return None

def get_flm_ensemble(distribution,process,decay_type,N_eachD,traj_lenth,dt,h,H,nu,mu,gamma,lamda,D_ensemble,alpha,D_beta,N_fbm_fixed,save=True):
    correm_size, eigenvalues = FBM.get_circulant_eigenvalue(decay_type, N_fbm_fixed, h, H, nu, mu)
    D_numbers=len(D_ensemble)
    N_traj=D_numbers*N_eachD
    trajectories = np.zeros((N_traj, traj_lenth + 1))
    for i in tqdm(range(D_numbers), desc=f"flm trajectory for H={H}"):
        for j in range(N_eachD):
            trajectories[i*N_eachD+j] = FLM.get_valid_FLM_trajectory(gamma, lamda, dt, traj_lenth, h, N_fbm_fixed,correm_size , D_ensemble[i]*eigenvalues)
    T = traj_lenth * dt
    ts = np.linspace(0, T, traj_lenth + 1)
    if save:
        files.save_trajectories_to_file(distribution,process,decay_type,T, dt, H, N_traj, nu, mu, alpha, D_beta, ts, trajectories,gamma)
    return None

if __name__ == "__main__":
    distribution,process,decay_type = sys.argv[1],sys.argv[2],sys.argv[3]
    fname=files.build_diffusivity_filename(distribution,alpha,D_beta,D_ensemble_num)
    D_ensemble,alpha,D_beta=files.read_diffusivity_from_data(fname)
    if process=='flm':
        for H in para.H_list:
            get_flm_ensemble(distribution,process, decay_type, N_eachD, traj_lenth, dt, h, H, nu, mu, gamma, lamda, D_ensemble,
                             alpha, D_beta, N_fbm_fixed)
    elif process=='fbm':
         for H in para.H_list:
             get_fbm_ensemble(distribution,process, decay_type, N_eachD, traj_lenth, dt, H, nu, mu, D_ensemble, alpha, D_beta,gamma)



