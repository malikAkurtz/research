import numpy as np
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