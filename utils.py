import numpy as np
from scipy.special import factorial
from scipy.constants import hbar, e
from scipy.special import hermite

PHI_0 = hbar / 2*e
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
    Takes in a wavefunction in the energy basis and 
    converts it to theta and phi to be plotted on the
    Bloch Sphere.
    """
    # Get amplitudes corresponding to E0 and E1    
    alpha = state_energy_basis[0]
    beta = state_energy_basis[1]
    
    rectangular_alpha = np.array([alpha.real, alpha.imag])
    rectangular_beta  = np.array([beta.real, beta.imag])
    
    polar_alpha = (np.sqrt(rectangular_alpha @ rectangular_alpha), np.arctan2(alpha.imag, alpha.real))
    polar_beta = (np.sqrt(rectangular_beta @ rectangular_beta), np.arctan2(beta.imag, beta.real))
    
    azimuth = polar_beta[1] - polar_alpha[1]
    inclanation = 2 * np.arccos(polar_alpha[0])
    
    return azimuth, inclanation

def spherical_to_rectangular(azimuth, inclanation):
    x = np.sin(inclanation) * np.cos(azimuth)
    y = np.sin(inclanation) * np.sin(azimuth)
    z = np.cos(inclanation)
    
    return x, y, z