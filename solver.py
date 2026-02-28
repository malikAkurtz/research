###############################################################################
#
#   solver.py
#
#   Crank-Nicolson time evolution, Rabi period estimation, unitary
#   construction, and gate fidelity for superconducting qubit circuits.
#
###############################################################################

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.constants import h, hbar
from scipy.signal import find_peaks

from quantization import QuantizationResult
from DriveParams import DriveParams


# =====================================================================
#   Result Container
# =====================================================================

@dataclass
class EvolutionResult:
    """Output of a single Crank-Nicolson run."""
    t_vec:     np.ndarray   # Time grid [s]
    At_vec:    np.ndarray   # Pulse envelope at each step [J]
    P_0:       np.ndarray   # Ground-state population vs time
    P_1:       np.ndarray   # First-excited-state population vs time
    P_2:       np.ndarray   # Second-excited-state population vs time (leakage)
    final_psi: np.ndarray   # Final state vector in the truncated energy basis


# =====================================================================
#   Drive Frequency Helper
# =====================================================================

def drive_params(energies: np.ndarray, detuning: float):
    """
    Compute drive frequency and period from qubit transition + detuning.

    Returns
    -------
    f01_Hz  : qubit transition frequency [Hz]
    f_drive : drive frequency [Hz]
    T_drive : drive period [s]
    """
    f0, f1  = energies[0:2] / h / 1e9   # [GHz]
    f01_Hz  = (f1 - f0) * 1e9            # [Hz]
    f_drive = f01_Hz + detuning
    T_drive = 1 / f_drive
    return f01_Hz, f_drive, T_drive


# =====================================================================
#   Crank-Nicolson Time Evolution
# =====================================================================

def crank_nicolson(
    quant: QuantizationResult,
    config: DriveParams,
    init_state: int,
    circuit=None,
    callback=None,
) -> EvolutionResult:
    """
    Time-evolve an initial energy eigenstate under an SFQ Gaussian pulse
    train using the Crank-Nicolson (implicit midpoint) method.

    Parameters
    ----------
    quant      : QuantizationResult from quantization.quantize()
    config     : DriveParams specifying the drive and evolution settings
    init_state : index of the initial eigenstate (0 = ground)
    circuit    : Circuit object — only needed when callback uses C or omega
    callback   : optional callable(step, basis, C, omega, states, t_vec,
                 P_0, P_1, P_2, psi) called every step (e.g. LivePlotter.update)

    Returns
    -------
    EvolutionResult
    """
    dim_sub           = config.dim_sub
    truncated_energies = quant.energies[:dim_sub]
    H_0 = np.diag(truncated_energies)

    if quant.basis == "charge":
        op = quant.n_hat_energy[:dim_sub, :dim_sub]
    else:
        op = quant.phi_hat_energy[:dim_sub, :dim_sub]

    _, _, T_drive = drive_params(quant.energies, config.detuning)

    f0, f1, f2 = quant.energies[0:3] / hbar   # [rad/s]
    alpha = (f2 - f1) - (f1 - f0)

    # Time grid
    T   = config.N_pulses * T_drive
    dt  = T_drive / config.steps_per_period
    N_t = round(T / dt)
    t_vec = np.arange(N_t) * dt

    # Pulse amplitude
    A_0 = config.amplitude_scale * (truncated_energies[1] - truncated_energies[0])

    # Pulse centers
    pulse_centers = np.arange(config.N_pulses) * T_drive

    # Initial state
    psi = np.zeros(dim_sub, dtype=complex)
    psi[init_state] = 1.0

    # Result arrays
    At_vec = np.zeros(N_t)
    P_0    = np.zeros(N_t)
    P_1    = np.zeros(N_t)
    P_2    = np.zeros(N_t)

    I = np.eye(dim_sub)

    # DRAG correction operator (σ_y-like in 1-2 subspace)
    n_12_y = np.zeros((dim_sub, dim_sub), dtype=complex)
    n_12_y[1, 2] = -1j * op[1, 2]
    n_12_y[2, 1] =  1j * op[2, 1]

    # Values needed only for the callback
    C     = circuit.capacitance_matrix[0][0] if circuit is not None else None
    omega = np.sqrt(circuit.omega_squared[0][0]) if circuit is not None else None

    for i in range(N_t):
        t_mid = t_vec[i] + dt / 2

        dt_to_pulses = t_mid - pulse_centers
        mask = np.abs(dt_to_pulses) < 4 * config.sigma

        gauss    = np.exp(-0.5 * (dt_to_pulses[mask] / config.sigma)**2)
        At       = A_0 * np.sum(gauss)
        At_dot   = A_0 * np.sum(-dt_to_pulses[mask] / config.sigma**2 * gauss)

        At_vec[i] = At

        drag_coeff = 0.0 if np.isclose(alpha, 0.0) else config.lambda_drag * At_dot / alpha
        H_mid = H_0 + At * op + drag_coeff * n_12_y

        A_mat = I + (1j * dt / (2 * hbar)) * H_mid
        B_mat = I - (1j * dt / (2 * hbar)) * H_mid
        psi   = np.linalg.solve(A_mat, B_mat @ psi)

        P_0[i] = abs(psi[0])**2
        P_1[i] = abs(psi[1])**2
        P_2[i] = abs(psi[2])**2

        if callback is not None:
            callback(i, quant.basis, C, omega, quant.states, t_vec, P_0, P_1, P_2, psi)

    return EvolutionResult(t_vec=t_vec, At_vec=At_vec, P_0=P_0, P_1=P_1, P_2=P_2, final_psi=psi)


# =====================================================================
#   Rabi Period Estimation
# =====================================================================

def calculate_rabi_period(
    quant: QuantizationResult,
    config: DriveParams,
    circuit=None,
    callback=None,
) -> tuple:
    """
    Run a long Crank-Nicolson simulation and extract the Rabi period
    from peaks in the smoothed P_1 oscillation.

    Returns
    -------
    rabi_period : float or None (None if fewer than 2 peaks found)
    result      : EvolutionResult from the run
    """
    result = crank_nicolson(quant, config, init_state=0, circuit=circuit, callback=callback)

    window    = 1000
    P1_smooth = np.convolve(result.P_1, np.ones(window) / window, mode='same')
    peaks, _  = find_peaks(P1_smooth, distance=10000)

    if len(peaks) >= 2:
        rabi_period = result.t_vec[peaks[1]] - result.t_vec[peaks[0]]
    else:
        rabi_period = None

    return rabi_period, result


# =====================================================================
#   Unitary Gate Construction
# =====================================================================

def build_unitary(
    d: int,
    quant: QuantizationResult,
    config: DriveParams,
    rabi_period: float,
) -> np.ndarray:
    """
    Construct the d-column unitary by evolving each computational basis
    state |0>, |1>, ..., |d-1> for a pi-pulse duration (half Rabi period).

    Returns
    -------
    U : ndarray, shape (dim_sub, d)
    """
    _, _, T_drive = drive_params(quant.energies, config.detuning)
    N_pi = round(rabi_period / 2 / T_drive)
    pi_config = replace(config, N_pulses=N_pi)

    columns = []
    for i in range(d):
        result = crank_nicolson(quant, pi_config, init_state=i)
        columns.append(result.final_psi)

    return np.array(columns).T


# =====================================================================
#   Gate Fidelity
# =====================================================================

def calculate_fidelity(
    U: np.ndarray,
    d: int,
    U_target: Optional[np.ndarray] = None,
) -> float:
    """
    Average gate fidelity:  F = |Tr(U_target† · U_actual)| / d

    Default target is the Pauli-X gate.
    """
    if U_target is None:
        U_target = np.array([[0, 1], [1, 0]], dtype=complex)
    return float(np.abs(np.trace(U_target.conj().T @ U[:d, :d])) / d)
