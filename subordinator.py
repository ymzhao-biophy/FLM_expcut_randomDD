import numpy as np
import parameter as para
step_number= 1000       #steps
def get_gamma(N,dt,gamma,lamda):
    shape_step=dt*gamma
    scale=1.0/lamda
    t=N*dt
    increments = np.random.gamma(shape=shape_step, scale=scale, size=N)
    traj = np.cumsum(increments)##sum the increments
    traj = np.concatenate(([0], traj))##add the original points
    tau = np.linspace(0, t, N + 1)
    return tau, traj
