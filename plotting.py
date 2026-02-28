###############################################################################
#
#   plotting.py
#
#   Visualization for superconducting circuit simulation results.
#   All functions accept plain data (arrays/scalars) rather than Circuit
#   objects, keeping plotting decoupled from simulation internals.
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e

from constants import PHI_0
from utils import charge_to_phase_basis, fock_to_phase_basis


# -------------------------------------------------------------------------
#   Charge-Basis Wavefunction Plot
# -------------------------------------------------------------------------

def plot_charge_distribution(n_cut: int, states: np.ndarray):
    """Plot |psi(n)|^2 for the three lowest eigenstates in the charge basis."""
    n_vals = np.arange(-n_cut, n_cut + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, np.abs(states[:, 0])**2, 'o-', label='|0>', markersize=4)
    plt.plot(n_vals, np.abs(states[:, 1])**2, 's-', label='|1>', markersize=4)
    plt.plot(n_vals, np.abs(states[:, 2])**2, '-',  label='|2>', markersize=4)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Number of Cooper Pairs (n)')
    plt.ylabel('Probability Amplitude')
    plt.title('Transmon Wavefunctions in Charge Basis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# -------------------------------------------------------------------------
#   Phase-Basis Wavefunction Plot
# -------------------------------------------------------------------------

def plot_phase_distribution(n_cut: int, states: np.ndarray, min_flux: float, max_flux: float, num_phases: int):
    """Fourier-transform charge-basis eigenstates to phase basis and plot."""
    phases = np.linspace(min_flux, max_flux, num_phases)
    states_phase = charge_to_phase_basis(states, n_cut, phases)

    plt.figure(figsize=(10, 6))
    plt.plot(phases, np.abs(states_phase[:, 0])**2, 'o-', label='|0>', markersize=4)
    plt.plot(phases, np.abs(states_phase[:, 1])**2, 's-', label='|1>', markersize=4)
    plt.plot(phases, np.abs(states_phase[:, 2])**2, '-',  label='|2>', markersize=4)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Phase')
    plt.ylabel('Probability Amplitude')
    plt.title('Transmon Wavefunctions in Phase Basis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# -------------------------------------------------------------------------
#   Potential Energy Landscape Plot
# -------------------------------------------------------------------------

def plot_potential_energy(circuit, energies: np.ndarray, min_flux: float, max_flux: float):
    """Plot the cosine potential V(phi) with energy eigenvalues overlaid."""
    node_phases = np.linspace(min_flux, max_flux, 400)
    U = np.array([circuit.get_potential_energy(np.array([phi * PHI_0])) for phi in node_phases])

    plt.plot(node_phases, U / e / 1e-3, 'r', linewidth=3, label='Potential Energy')
    for k in range(6):
        plt.hlines(energies[k] / e / 1e-3, xmin=min_flux, xmax=max_flux,
                   colors='k', linewidth=2, label="Energy Level" if k == 0 else None)
    plt.xlabel(r'Phase $\phi$', fontsize=15)
    plt.ylabel('Energy (meV)', fontsize=15)
    plt.axis([min_flux, max_flux, None, None])
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.grid(alpha=0.3)
    plt.show()


# -------------------------------------------------------------------------
#   SFQ Pulse Sequence Plot
# -------------------------------------------------------------------------

def plot_pulse_sequence(t_vec: np.ndarray, At_vec: np.ndarray, num_steps: int):
    """Plot the first num_steps of the SFQ Gaussian pulse train envelope."""
    plt.figure(figsize=(8, 4))
    plt.plot(t_vec[:num_steps] * 1e9, At_vec[:num_steps] / e / 1e-3, linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Pulse (meV)')
    plt.title('SFQ Pulse Shape')
    plt.grid(True)
    plt.show()


# -------------------------------------------------------------------------
#   Rabi Oscillation Plot
# -------------------------------------------------------------------------

def plot_rabi_oscillations(f_drive: float, t_vec: np.ndarray, P_0: np.ndarray, P_1: np.ndarray, P_2: np.ndarray):
    """Plot population dynamics (P0, P1, P2) vs time showing Rabi oscillations."""
    plt.figure(figsize=(8, 5))
    plt.plot(t_vec * 1e9, P_0, label='P0', linewidth=2)
    plt.plot(t_vec * 1e9, P_1, label='P1', linewidth=2)
    plt.plot(t_vec * 1e9, P_2, label='P2 (leakage)', linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Population')
    plt.title(f'SFQ-driven Rabi dynamics: f_drive = {f_drive / 1e9:.3f} GHz')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# -------------------------------------------------------------------------
#   Combined 2x3 Dashboard Plot
# -------------------------------------------------------------------------

def plot_all(circuit, quant, result, n_cut, min_flux, max_flux, num_phases, f_drive):
    """
    Generate a 2x3 subplot dashboard showing:
      (0,0) Charge-basis wavefunctions
      (0,1) Phase-basis wavefunctions
      (0,2) Potential energy landscape with energy levels
      (1,0) SFQ pulse shape (first 1000 steps)
      (1,1) Rabi oscillation dynamics
      (1,2) (empty)

    Parameters
    ----------
    circuit    : Circuit  — for get_potential_energy()
    quant      : QuantizationResult
    result     : EvolutionResult
    n_cut      : int
    min_flux   : float  [rad]
    max_flux   : float  [rad]
    num_phases : int
    f_drive    : float  [Hz]
    """
    _fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # ---- (0,0) Wavefunction in original basis ----
    if quant.basis == "charge":
        n_vals = np.arange(-n_cut, n_cut + 1)
        basis_xlabel = 'Number of Cooper Pairs (n)'
        basis_title  = 'Wavefunctions in Charge Basis'
    else:
        n_vals = np.arange(n_cut)
        basis_xlabel = 'Fock State (n)'
        basis_title  = 'Wavefunctions in Fock Basis'
    axes[0, 0].plot(n_vals, np.abs(quant.states[:, 0])**2, 'o-', label='|0>', markersize=4)
    axes[0, 0].plot(n_vals, np.abs(quant.states[:, 1])**2, 's-', label='|1>', markersize=4)
    axes[0, 0].plot(n_vals, np.abs(quant.states[:, 2])**2, '-',  label='|2>', markersize=4)
    axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel(basis_xlabel)
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_title(basis_title)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ---- (0,1) Phase Distribution ----
    phases = np.linspace(min_flux, max_flux, num_phases)
    if quant.basis == "charge":
        states_phase = charge_to_phase_basis(quant.states, n_cut, phases)
    else:
        C     = circuit.capacitance_matrix[0][0]
        omega = np.sqrt(circuit.omega_squared[0][0])
        states_phase = np.column_stack([
            fock_to_phase_basis(quant.states[:, k], C, omega, n_cut, phases)
            for k in range(3)
        ])
    axes[0, 1].plot(phases, np.abs(states_phase[:, 0])**2, 'o-', label='|0>', markersize=4)
    axes[0, 1].plot(phases, np.abs(states_phase[:, 1])**2, 's-', label='|1>', markersize=4)
    axes[0, 1].plot(phases, np.abs(states_phase[:, 2])**2, '-',  label='|2>', markersize=4)
    axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Phase')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes[0, 1].set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    axes[0, 1].set_title('Wavefunctions in Phase Basis')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ---- (0,2) Potential Energy ----
    node_phases = np.linspace(min_flux, max_flux, 400)
    U = np.array([circuit.get_potential_energy(np.array([phi * PHI_0])) for phi in node_phases])
    axes[0, 2].plot(node_phases, U / e / 1e-3, 'r', linewidth=3, label='Potential Energy')
    for k in range(6):
        axes[0, 2].hlines(quant.energies[k] / e / 1e-3, xmin=min_flux, xmax=max_flux,
                          colors='k', linewidth=2, label="Energy Level" if k == 0 else None)
    axes[0, 2].set_xlabel(r'Phase $\phi$')
    axes[0, 2].set_ylabel('Energy (meV)')
    axes[0, 2].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes[0, 2].set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    axes[0, 2].set_title('Potential Energy')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # ---- (1,0) Pulse Sequence ----
    axes[1, 0].plot(result.t_vec * 1e9, result.At_vec / e / 1e-3, linewidth=2)
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('Pulse (meV)')
    axes[1, 0].set_title('SFQ Pulse Shape')
    axes[1, 0].grid(True)

    # ---- (1,1) Rabi Oscillations ----
    axes[1, 1].plot(result.t_vec * 1e9, result.P_0, label='P0', linewidth=2)
    axes[1, 1].plot(result.t_vec * 1e9, result.P_1, label='P1', linewidth=2)
    axes[1, 1].plot(result.t_vec * 1e9, result.P_2, label='P2 (leakage)', linewidth=2)
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].set_ylabel('Population')
    axes[1, 1].set_title(f'SFQ-driven Rabi: f_drive = {f_drive / 1e9:.3f} GHz')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True)

    # ---- (1,2) Empty ----
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig("simulation_results.png")
