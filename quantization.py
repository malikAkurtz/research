###############################################################################
#
#   quantization.py
#
#   Builds the quantum Hamiltonian in the charge or Fock basis, diagonalizes
#   it, and transforms operators into the energy eigenbasis.
#
###############################################################################

from dataclasses import dataclass
from typing import Optional

import numpy as np
from constants import *

from Circuit import Circuit


@dataclass
class QuantizationResult:
    """All quantum data produced after quantization + diagonalization."""
    basis: str                              # "charge" or "fock"
    n_cut: int                              # Hilbert-space truncation
    n_hat: np.ndarray                       # Charge/number operator (original basis)
    H_hat: np.ndarray                       # Hamiltonian (original basis)
    phi_hat: Optional[np.ndarray]           # Phase operator (fock basis only)
    energies: np.ndarray                    # Energy eigenvalues [J]
    states: np.ndarray                      # Eigenvectors (columns)
    n_hat_energy: Optional[np.ndarray]      # Charge operator in energy eigenbasis
    phi_hat_energy: Optional[np.ndarray]    # Phase operator in energy eigenbasis


def quantize(circuit: Circuit, n_cut: int) -> QuantizationResult:
    """
    Build the charge-basis (or Fock-basis) Hamiltonian, diagonalize it,
    and transform the coupling operator into the energy eigenbasis.

    Parameters
    ----------
    circuit : Circuit
        A fully constructed Circuit object.
    n_cut : int
        Hilbert-space truncation: Cooper-pair numbers run from -n_cut to +n_cut
        (charge basis), or Fock states 0..n_cut-1 (Fock basis).

    Returns
    -------
    QuantizationResult
    """
    if circuit.N != 1:
        raise NotImplementedError("Quantization currently supports single-node circuits only")

    C  = circuit.capacitance_matrix[0][0]
    EC = e**2 / (2 * C)

    phi_hat       = None
    n_hat_energy  = None
    phi_hat_energy = None

    if len(circuit.josephson_elements) > 0:
        # ------------------------------------------------------------------
        #   Charge basis:  H = 4 EC n^2 - (EJ/2)(|n><n+1| + h.c.)
        # ------------------------------------------------------------------
        basis = "charge"
        EJ    = circuit.josephson_elements[0].EJ
        n_vals = np.arange(-n_cut, n_cut + 1)
        dim    = len(n_vals)
        n_hat  = np.diag(n_vals.astype(float))
        H = (4 * EC * (n_hat @ n_hat)) - 0.5 * EJ * (
            np.diag(np.ones(dim - 1), 1) + np.diag(np.ones(dim - 1), -1)
        )

    else:
        # ------------------------------------------------------------------
        #   Fock basis:  H = Q^2/(2C) + phi^2/(2L)
        # ------------------------------------------------------------------
        basis   = "fock"
        omega   = np.sqrt(circuit.omega_squared[0][0])
        L       = 1 / circuit.inv_inductance_matrix[0][0]

        a_plus  = np.zeros((n_cut, n_cut))
        a_minus = np.zeros((n_cut, n_cut))
        for m in range(n_cut):
            for n in range(n_cut):
                if m == n + 1:
                    a_plus[m][n]  = np.sqrt(n + 1)
                elif m == n - 1:
                    a_minus[m][n] = np.sqrt(n)

        phi_zpf = np.sqrt(2 * hbar * C * omega) / (2 * C * omega * PHI_0)
        Q_zpf   = np.sqrt(2 * hbar * C * omega)
        
        Q_hat   = (Q_zpf / (2 * 1j)) * (a_minus - a_plus)
        phi_hat = phi_zpf * (a_minus + a_plus) 
        
        PHI_hat = phi_hat * PHI_0
        H       = (Q_hat @ Q_hat) / (2 * C) + (PHI_hat @ PHI_hat) / (2 * L)
        n_hat   = a_plus @ a_minus

    # Diagonalize
    energies, states = np.linalg.eigh(H)

    # Change basis
    if basis == "charge":
        n_hat_energy = np.conjugate(states).T @ n_hat @ states
    else:
        phi_hat_energy = np.conjugate(states).T @ phi_hat @ states
        n_hat_energy = np.conjugate(states).T @ n_hat @ states

    return QuantizationResult(
        basis=basis,
        n_cut=n_cut,
        n_hat=n_hat,
        H_hat=H,
        phi_hat=phi_hat,
        energies=energies,
        states=states,
        n_hat_energy=n_hat_energy,
        phi_hat_energy=phi_hat_energy,
    )
