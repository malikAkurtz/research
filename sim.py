import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, hbar, e

# Transmon + SFQ pulse train + Crank–Nicolson Rabi simulation (AG)

# Transmon parameters
fC        = 250e6           # Charging energy frequency EC/h [Hz]
EC        = h * fC          # EC in Joules
EJ_EC     = 50              # EJ/EC ratio (typical transmon)
EJ        = EJ_EC * EC      # Josephson energy [J]

# Charge basis (|n> states)
n_cut = 20                                  # max |n| to keep
n     = [i for i in range(-n_cut, n_cut+1)] # e.g. [-20, -19, ..., 0, ..., 19, 20] for n_cut = 20
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
        
print(f"n_hat: \n{n_hat}")
print(f"n_hat.shape: {n_hat.shape}")

print(f"H: \n{H}")
print(f"H.shape: {H.shape}")

# Diagonalize H (eigh returns eigenvalues/vectors in sorted order)
evals, V = np.linalg.eigh(H)
D = np.diag(evals)

# Change basis of n_hat to eigenbasis (energy basis)
n_op = V.T @ n_hat @ V

print(f"D: \n{D}")
print(f"D.shape: {D.shape}")

print(f"V: \n{V}")
print(f"V.shape: {V.shape}")

# Superconducting phase
phi = np.linspace(-np.pi, np.pi, 400)

# Josephson potential [J]
U = -EJ * np.cos(phi)

# Plot Josephson potential
plt.plot(phi, U / q / 1e-3, 'r', linewidth=3, label='Josephson potential')

# Overlay lowest energy eigenvalues
for k in range(6):  # ground, first, second excited
    plt.hlines(evals[k] / q / 1e-3, xmin=-np.pi, xmax=np.pi,
               colors='k', linewidth=2, linestyles='-', label="Energy Level" if k == 0 else None)

# Labels and formatting
plt.xlabel('φ', fontsize=15)
plt.ylabel('Energy (meV)', fontsize=15)
plt.axis([-np.pi, np.pi, -0.06, 0.05])
plt.xticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])
plt.legend(fontsize=12)
plt.show()

# Basic spectroscopic info
f01 = (evals[1] - evals[0]) / h / 1e9   # |0> -> |1> [GHz]
f12 = (evals[2] - evals[1]) / h / 1e9   # |1> -> |2> [GHz]
anh = f12 - f01                          # anharmonicity [GHz]

print("Transmon parameters:")
print(f"  EC/h  = {fC / 1e9:.3f} GHz")
print(f"  EJ/EC = {EJ_EC:.1f}")
print(f"  f01   = {f01:.3f} GHz")
print(f"  f12   = {f12:.3f} GHz")
print(f"  Anharmonicity (f12 - f01) = {anh:.3f} GHz\n")

# Choose a truncated eigenbasis (low levels only)
dim_sub = 6

evals = evals[:dim_sub]
H0 = np.diag(evals)

# Truncate n_hat
n_op = n_op[:dim_sub, :dim_sub]


# Drive frequency and SFQ pulse parameters
f01_Hz   = f01 * 1e9             # |0>->|1| in Hz
delta_f  = -1e6                  # detuning (Hz)
f_drive  = f01_Hz + delta_f      # drive repetition rate ~ resonant with 0-1 + detuning
T_drive  = 1 / f_drive           # pulse repetition period [s]


Npulses  = 1000                  # number of SFQ pulses in the train
t_total  = Npulses * T_drive     # total simulation time

sigma    = 15e-12                     # SFQ pulse width (std dev) ~ 2-20 ps
A0       = 0.01 * (evals[1]-evals[0]) # Pulse amplitude in Joules 

# Time discretization and Crank-Nicolson setup
steps_per_period = 200          # time steps per drive period
dt       = T_drive / steps_per_period
Nt       = round(t_total / dt)
t_vec    = np.array([i for i in range(Nt)]).T * dt

# Precompute pulse centers
pulse_centers = np.array([i for i in range(Npulses)]) * T_drive;

# Initial state: ground state |0> in eigenbasis
psi = np.zeros(dim_sub);
psi[0] = 1.0;

# Pre-allocate arrays for results
At = np.zeros(Nt)
P0 = np.zeros(Nt)
P1 = np.zeros(Nt)
P2 = np.zeros(Nt)

# Helper identity matrix
I_sub = np.eye(dim_sub)

# Time evolution via Crank–Nicolson
for it in range(Nt):
    t = t_vec[it]
    tm = t + dt / 2  # mid-point time for CN Hamiltonian [cite: 294]

    # SFQ pulse train envelope: sum of Gaussians centered at pulse_centers
    # Only sum nearby pulses (within ~4 sigma) for efficiency
    dt_to_pulses = tm - pulse_centers
    mask = np.abs(dt_to_pulses) < 4 * sigma
    A_t = np.sum(A0 * np.exp(-0.5 * (dt_to_pulses[mask] / sigma)**2))
    At[it] = A_t

    # Total Hamiltonian in eigenbasis at mid-time [cite: 114, 284, 294]
    H_mid = H0 + A_t * n_op

    # Crank–Nicolson update matrices:
    # (I + i*dt/(2*hbar) * H_mid) * psi_next = (I - i*dt/(2*hbar) * H_mid) * psi_now
    A_mat = I_sub + (1j * dt / (2 * h_bar)) * H_mid
    B_mat = I_sub - (1j * dt / (2 * h_bar)) * H_mid

    # Solve the linear system A_mat * psi = B_mat * psi_old
    psi = np.linalg.solve(A_mat, B_mat @ psi)

    # Normalize to prevent numerical drift
    psi = psi / np.linalg.norm(psi)

    # Store populations in |0>, |1>, |2> [cite: 290, 291]
    P0[it] = np.abs(psi[0])**2
    P1[it] = np.abs(psi[1])**2
    P2[it] = np.abs(psi[2])**2

# --- Plotting ---

# Figure 2: Pulse sequence (First 1000 steps)
plt.figure(figsize=(8, 4))
plt.plot(t_vec[:1000] * 1e9, At[:1000] / q / 1e-3, linewidth=2)
plt.xlabel('Time (ns)')
plt.ylabel('Pulse (meV)')
plt.title('SFQ Pulse shape')
plt.grid(True)
plt.show()

# Figure 3: Populations (Rabi oscillations + leakage)
plt.figure(figsize=(8, 5))
plt.plot(t_vec * 1e9, P0, label='P0', linewidth=2)
plt.plot(t_vec * 1e9, P1, label='P1', linewidth=2)
plt.plot(t_vec * 1e9, P2, label='P2 (leakage)', linewidth=2)
plt.xlabel('Time (ns)')
plt.ylabel('Population')
plt.title(f'SFQ-driven Rabi dynamics: f_drive = {f_drive/1e9:.3f} GHz')
plt.legend(loc='best')
plt.grid(True)
plt.show()
