import numpy as np
np.set_printoptions(precision=8, linewidth=120, suppress=True)

from dataclasses import replace
from scipy.constants import h, e
import matplotlib.pyplot as plt

from Circuit import Circuit
from quantization import quantize, QuantizationResult
from solver import calculate_rabi_period, build_unitary, calculate_fidelity
from DriveParams import DriveParams
from config import *


def parameter_sweep(quant: QuantizationResult, base_config: DriveParams, param_name: str,
                    min_param: float, max_param: float, resolution: int):
    """
    Sweep one DriveParams field over a range and compute fidelity at each value.

    Parameters
    ----------
    quant       : QuantizationResult from quantize()
    base_config : DriveParams to use as the baseline
    param_name  : name of the DriveParams field to sweep (e.g. 'lambda_drag')
    min_param   : minimum parameter value
    max_param   : maximum parameter value
    resolution  : number of points in the sweep

    Returns
    -------
    params     : swept parameter values
    fidelities : fidelity at each value
    """
    params     = np.linspace(min_param, max_param, resolution)
    fidelities = []

    for val in params:
        config = replace(base_config, **{param_name: val})
        rabi_period, _ = calculate_rabi_period(quant, config)

        if rabi_period is None:
            fidelities.append(0.0)
            continue

        U        = build_unitary(d=2, quant=quant, config=config, rabi_period=rabi_period)
        fidelity = calculate_fidelity(U, d=2)
        fidelities.append(fidelity)

    return params, fidelities


def main():
    # -------------------------------------------------------------------------
    #   Hyper-Parameters
    # -------------------------------------------------------------------------

    fC    = 250e6
    EJ_EC = 50
    n_cut = 20

    base_config = DriveParams(
        dim_sub          = 6,
        detuning         = OPTIMAL_DETUNING,
        N_pulses         = 1000,
        amplitude_scale  = OPTIMAL_AMPLITUDE_SCALE,
        sigma            = OPTIMAL_SIGMA,
        lambda_drag      = OPTIMAL_LAMBDA_DRAG,
        steps_per_period = 200,
    )

    # -------------------------------------------------------------------------
    #   Circuit & Quantization
    # -------------------------------------------------------------------------

    EC    = h * fC
    C_sum = (e**2) / (2 * EC)
    CJ    = 0.05 * C_sum
    CS    = 0.95 * C_sum
    EJ    = EJ_EC * EC

    graph_rep = {
        'nodes': ['a'],
        'capacitors': [('a', 'gnd', CS), ('a', 'gnd', CJ)],
        'inductors': [],
        'josephson_elements': [('a', 'gnd', EJ)],
        'external_flux': {}
    }

    circuit = Circuit(graph_rep=graph_rep)
    quant   = quantize(circuit, n_cut=n_cut)

    # -------------------------------------------------------------------------
    #   Parameter Sweep
    # -------------------------------------------------------------------------

    params, fidelities = parameter_sweep(
        quant      = quant,
        base_config = base_config,
        param_name  = 'lambda_drag',
        min_param   = 0.0,
        max_param   = 0.02,
        resolution  = 21,
    )

    print(f"Best lambda_drag: {params[np.argmax(fidelities)]:.4f}  (fidelity = {max(fidelities):.4f})")

    plt.figure(figsize=(8, 4))
    plt.plot(params, fidelities, linewidth=2)
    plt.xlabel('Parameter Value')
    plt.ylabel('Fidelity')
    plt.title('Fidelity vs Parameter Value')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
