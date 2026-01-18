import numpy as np

# Transmon + SFQ pulse train + Crank–Nicolson Rabi simulation (AG)

# Physical constants
h       = 6.62607015e-34      # Planck constant [J·s]
h_bar   = h/(2*np.pi)
q       =1.6e-19;             # electron chrage in Coulombs 


# Transmon parameters
fC        = 250e6           # Charging energy frequency EC/h [Hz]
EC        = h * fC          # EC in Joules
EJ_EC     = 50              # EJ/EC ratio (typical transmon)
EJ        = EJ_EC * EC      # Josephson energy [J]

# Charge basis (|n> states)
n_cut = 20                                  # max |n| to keep
n     = [i for i in range(-n_cut, n_cut+1)] # integer charge states
N     = len(n)                              # Hilbert space dimension
print(f"N: {N}")
# Create Hamiltonain
H = np.zeros((N, N))

# Create charge operator in charge basis
n_hat = np.zeros((N, N))

# Populate operators
for i in range(N):
    for j in range(N):
        if i == j:
            charge       = i - n_cut
            H[i][j]      = 4*EC * (charge**2)
            n_hat[i][j]  = charge
        elif np.abs(i-j) == 1:
            H[i][j] = -EJ/2
        else:
            continue

# Diagonalize H
lamda, V = np.linalg.eigh(H)
lamda = np.diag(lamda)

# Change basis of n_hat to eigenbasis
n_op = V.T @ n_hat @ V


