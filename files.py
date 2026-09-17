import os
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)
import numpy as np

def build_diffusivity_filename(distribution,alpha,D_beta,N):
    return f"D_ensmble_of_{distribution}_distribution_alpha{alpha}_Dbeta{D_beta}_N{N}.npz"

def build_traj_ensemble_filename(distribution,process,decay_tipe, T, dt, H, N_traj, nu, mu, alpha, D_beta,gamma):
    return f"{distribution}D_{process}_{decay_tipe}_decay_T{T:.1f}_gamma{gamma:.3f}_dt{dt:.2f}_H{H:.3f}_N{int(N_traj)}_nu{nu:.2f}_mu{mu:.1f}_alpha{alpha:.2f}_Dbeta{D_beta:.2f}.npz"

def save_diffusivity_to_file(distribution,alpha,D_beta,N,D_ensemble):
    fname = build_diffusivity_filename(distribution,alpha, D_beta, N)
    path = f"{save_dir}/{fname}"
    np.savez(path, alpha=alpha,D_beta=D_beta,D_ensemble=D_ensemble)
    print(f"Diffusivity data save to {path}")

def read_diffusivity_from_data(fname):
    filepath = os.path.join(save_dir, fname)
    data = np.load(filepath)
    alpha = data['alpha'].item()
    D_beta = data['D_beta'].item()
    D_ensemble= data['D_ensemble']
    return D_ensemble,alpha,D_beta

def save_trajectories_to_file(distribution,process,decay_tipe, T, dt, H, N_traj, nu, mu, alpha, D_beta, ts, trajectories,gamma):
    fname = build_traj_ensemble_filename(distribution,process,decay_tipe, T, dt, H, N_traj, nu, mu, alpha, D_beta,gamma)
    path = f"{save_dir}/{fname}"
    np.savez(path, ts=ts,trajectories=trajectories)

def read_trajectories_data(fname):
    filepath = os.path.join(save_dir, fname)
    data = np.load(filepath)
    ts=data['ts']
    trajectories=data['trajectories']
    return ts,trajectories


