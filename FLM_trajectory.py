import numpy as np
import FBM_trajectory as FBM
import subordinator as sub
import parameter as para
import sys

def get_FLM_trajectory(gamma,lamda,dt,N,h,N_fbm_fixed,correm_size,eigenvalues):
    ts, gamma_traj = sub.get_gamma(N, dt, gamma, lamda)
    if gamma_traj[-1]>N_fbm_fixed*h:
        print("warning_length_too_long")
        return None
    N_fbm = int(gamma_traj[-1]/h)+1
    fbm_traj = FBM.generate_FBM_trajectory(correm_size, eigenvalues, N_fbm)
    hs = np.arange(len(fbm_traj)) * h
    real_traj = np.interp(gamma_traj, hs, fbm_traj)  ##insert the gamma subordinator into FBM
    return real_traj

def get_valid_FLM_trajectory(gamma, lamda, dt, N, h,N_fbm_fixed,correm_size , eigenvalues, max_attempts=100):
    for attempt in range(max_attempts):
        result = get_FLM_trajectory(gamma, lamda, dt, N, h, N_fbm_fixed,correm_size, eigenvalues)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid FLM trajectory after {max_attempts} attempts.")






