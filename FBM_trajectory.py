import numpy as np

def get_circulant_eigenvalue(decay_tpye,N,h,H,nu,mu):
    ##get the dimension of the circulant matrix
    v = int(np.ceil(np.log2(N)))
    m = 2 * (2 ** v)
    ##define the circulant matrix
    c = np.zeros(m)
    for j in range(m):
        lag = j if j <= m // 2 else m - j
        if decay_tpye=='power':
            decay=1/(1+lag*h*nu)**mu
        elif decay_tpye=='exponential':
            decay=np.exp(-nu*lag*h)
        elif decay_tpye=='no':
            decay=1
        c[j] = decay*(0.5 * (h ** (2 * H)) *
        (
                np.abs(lag - 1) ** (2 * H)
                +np.abs(lag + 1) ** (2 * H)
                -2 * np.abs(lag) ** (2 * H)
        ))
    lamda = np.real(np.fft.fft(c)) ## get eigenvalues by fft
    if np.any(lamda < -1e-6):
        print("Warning: negative eigenvalues detected")
        lamda = np.abs(lamda)

    return m, lamda


def generate_FBM_trajectory(m,lamda,N):
    half_m = m // 2
    a = np.zeros(m, dtype=complex)

    U = np.random.randn(half_m + 1)
    V = np.random.randn(half_m + 1)

    for j in range(half_m + 1):
        if j == 0 or j == half_m:
            a[j] = np.sqrt(lamda[j]) * np.random.randn() / np.sqrt(m)
        else:
            a[j] = np.sqrt(lamda[j]) / np.sqrt(2 * m) * (U[j] + 1j * V[j])
            a[m - j] = np.sqrt(lamda[j]) / np.sqrt(2 * m) * (U[j] - 1j * V[j])

    g = np.fft.fft(a)
    noise = np.real(g[:N])##getting the noise
    trajectory=np.cumsum(noise)
    trajectory=np.insert(trajectory,0,0)##inset 0 as the initial state
    return trajectory






