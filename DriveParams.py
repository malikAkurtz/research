from dataclasses import dataclass


@dataclass
class DriveParams:
    """Parameters that define the SFQ drive and time-evolution settings."""
    dim_sub: int            # Number of energy levels to keep in truncated Hilbert space
    detuning: float         # Drive frequency offset from f_01 [Hz]
    N_pulses: int           # Total number of SFQ pulses
    amplitude_scale: float  # Pulse amplitude as fraction of (E1 - E0)
    sigma: float            # Gaussian pulse width [s]
    lambda_drag: float      # DRAG correction coefficient
    steps_per_period: int   # Time steps per drive period
