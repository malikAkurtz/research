###############################################################################
#
#   non_linear.py
#
#   Simulates a Cooper Pair Box / Transmon-like superconducting circuit
#   (single Josephson junction with shunt capacitance). Orchestrates the
#   full pipeline: circuit construction → quantization → Rabi period →
#   unitary construction → gate fidelity.
#
###############################################################################

import numpy as np
np.set_printoptions(precision=8, linewidth=120, suppress=True)

from scipy.constants import h, e

from Circuit import Circuit
from quantization import quantize
from solver import drive_params, calculate_rabi_period, build_unitary, calculate_fidelity
from DriveParams import DriveParams
from plotting import plot_all
from LivePlotter import LivePlotter
from config import *


def main():
    # -------------------------------------------------------------------------
    #   Hyper-Parameters
    # -------------------------------------------------------------------------

    fC    = 250e6   # Charging energy frequency EC/h [Hz]
    EJ_EC = 50      # EJ/EC ratio (typical transmon)
    n_cut = 20      # Charge-basis truncation

    drive_config = DriveParams(
        dim_sub          = 6,
        detuning         = OPTIMAL_DETUNING,
        N_pulses         = 1000,
        amplitude_scale  = OPTIMAL_AMPLITUDE_SCALE,
        sigma            = OPTIMAL_SIGMA,
        lambda_drag      = OPTIMAL_LAMBDA_DRAG,
        steps_per_period = 200,
    )

    # -------------------------------------------------------------------------
    #   Derived Physical Constants
    # -------------------------------------------------------------------------

    EC    = h * fC
    C_sum = (e**2) / (2 * EC)
    CJ    = 0.05 * C_sum
    CS    = 0.95 * C_sum
    EJ    = EJ_EC * EC

    # -------------------------------------------------------------------------
    #   Circuit Graph Representation
    # -------------------------------------------------------------------------

    graph_rep = {
        'nodes': ['a'],
        'capacitors': [
            ('a', 'gnd', CS),
            ('a', 'gnd', CJ),
        ],
        'inductors': [],
        'josephson_elements': [
            ('a', 'gnd', EJ),
        ],
        'external_flux': {}
    }

    # -------------------------------------------------------------------------
    #   Circuit Construction & Topology Info
    # -------------------------------------------------------------------------

    circuit = Circuit(graph_rep=graph_rep)

    for node in circuit.active_nodes:
        print(f"  Active node: {node.label}")
    for node in circuit.passive_nodes:
        print(f"  Passive node: {node.label}")

    print(circuit)
    print("Capacitance Matrix [fF]:")
    print(circuit.capacitance_matrix * 1e15)
    print("Inverse Inductance Matrix [1/nH]:")
    print(circuit.inv_inductance_matrix / 1e9)

    # -------------------------------------------------------------------------
    #   Step 1: Quantize
    # -------------------------------------------------------------------------

    quant = quantize(circuit, n_cut=n_cut)
    print("Quantization + Diagonalization + Basis Change Complete.")
    print("Charge Operator (n_hat) [dimensionless]:")
    print(quant.n_hat)
    print("Hamiltonian in Charge Basis (H_hat) [GHz]:")
    print(quant.H_hat / (h * 1e9))
    print("Eigenvalues [GHz]:")
    print(quant.energies / (h * 1e9))
    print("Charge Operator in Energy Basis:")
    print(quant.n_hat_energy)
    print(f"# Energy Levels: {quant.states.shape[0]}")

    # -------------------------------------------------------------------------
    #   Transmon Parameter Summary
    # -------------------------------------------------------------------------

    f0, f1, f2 = quant.energies[0:3] / h / 1e9
    alpha = (f2 - f1) - (f1 - f0)

    print("Transmon parameters:")
    print(f"  EC/h          = {EC / (h * 1e9):.3f} GHz")
    print(f"  EJ/EC         = {EJ / EC:.1f}")
    print(f"  Anharmonicity = {alpha:.3f} GHz")

    # -------------------------------------------------------------------------
    #   Step 2: Rabi Period
    # -------------------------------------------------------------------------

    _, f_drive, _ = drive_params(quant.energies, drive_config.detuning)

    if LIVE_VISUALIZATION:
        live_plotter = LivePlotter(
            basis           = quant.basis,
            f_drive         = f_drive,
            dim_sub         = drive_config.dim_sub,
            n_cut           = n_cut,
            min_flux        = -np.pi,
            max_flux        = np.pi,
            update_interval = 100,
        )
        rabi_period, rabi_result = calculate_rabi_period(
            quant, drive_config, circuit=circuit, callback=live_plotter.update
        )
        live_plotter.finalize()
    else:
        rabi_period, rabi_result = calculate_rabi_period(
            quant, drive_config, circuit=circuit
        )

    plot_all(
        circuit    = circuit,
        quant      = quant,
        result     = rabi_result,
        n_cut      = n_cut,
        min_flux   = -np.pi,
        max_flux   = np.pi,
        num_phases = 1000,
        f_drive    = f_drive,
    )

    print(f"Rabi Period: {rabi_period * 1e9:.2f} ns")

    # -------------------------------------------------------------------------
    #   Step 3: Build Unitary & Evaluate Fidelity
    # -------------------------------------------------------------------------

    d = 2
    U = build_unitary(d=d, quant=quant, config=drive_config, rabi_period=rabi_period)
    fidelity = calculate_fidelity(U, d=d)

    print("Unitary Operator:")
    print(U)
    print(f"Top-left {d}x{d}:")
    print(U[:d, :d])
    print(f"Gate Fidelity: {fidelity}")


if __name__ == "__main__":
    main()
