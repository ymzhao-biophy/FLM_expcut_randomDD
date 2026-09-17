import numpy as np
import files
import parameter as para
import sys
def generate_gamma_diffusivity(distribution,alpha,D_beta,N_D):
    if distribution=='gamma':
        D_ensemble = np.random.gamma(shape=alpha, scale=D_beta, size=N_D)
    elif distribution=='const':
        D_ensemble = np.ones(N_D)*D_beta
    else:
        print('no such a choice')
        return 0
    files.save_diffusivity_to_file(distribution, alpha , D_beta, N_D, D_ensemble)
    return 0

if __name__ == "__main__":
    distribution=sys.argv[1]
    generate_gamma_diffusivity(distribution,para.alpha,para.D_beta,para.D_ensemble_num)




