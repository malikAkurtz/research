import numpy as np
from scipy.special import factorial
from scipy.constants import hbar
from scipy.special import hermite

from constants import PHI_0

# -------------------------------------------------------------------------
#   Charge-to-Phase Basis Transform (Vectorized)
# -------------------------------------------------------------------------

def charge_to_phase_basis(states, n_cut, phases):
    """
    Vectorized discrete Fourier transform from charge basis to phase basis.

    Parameters
    ----------
    states : ndarray, shape (2*n_cut+1, num_states)
        Eigenstates in the charge basis (columns = states).
    n_cut  : int
        Charge truncation parameter.
    phases : ndarray, shape (num_phases,)
        Phase values at which to evaluate the wavefunctions.

    Returns
    -------
    ndarray, shape (num_phases, num_states)
        States expressed in the phase basis.
    """
    n_vals = np.arange(-n_cut, n_cut + 1)
    E = np.exp(1j * np.outer(phases, n_vals))
    return E @ states


def fock_to_phase_basis(psi_fock, C, omega, n_cut, phases):
    xi = np.sqrt(C * omega / hbar) * PHI_0 * phases
    prefactor = (C * omega / (np.pi * hbar)) ** 0.25

    basis = np.zeros((len(phases), n_cut))
    for n in range(n_cut):
        Hn = hermite(n)
        norm = 1.0 / np.sqrt(2**n * factorial(n))
        basis[:, n] = prefactor * norm * Hn(xi) * np.exp(-xi**2 / 2)

    return basis @ psi_fock


def spherical_coords(state_energy_basis):
    """
    Converts a wavefunction in the energy basis to (azimuth, inclination)
    for plotting on the Bloch sphere.
    """
    alpha = state_energy_basis[0]
    beta  = state_energy_basis[1]

    r_alpha = np.array([alpha.real, alpha.imag])

    mag_alpha = np.sqrt(r_alpha @ r_alpha)

    azimuth     = np.arctan2(beta.imag, beta.real) - np.arctan2(alpha.imag, alpha.real)
    inclination = 2 * np.arccos(np.clip(mag_alpha, 0, 1))

    return azimuth, inclination


def spherical_to_rectangular(azimuth, inclination):
    x = np.sin(inclination) * np.cos(azimuth)
    y = np.sin(inclination) * np.sin(azimuth)
    z = np.cos(inclination)
    return x, y, z
