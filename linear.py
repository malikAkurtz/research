import numpy as np
np.set_printoptions(precision=8, linewidth=120, suppress=True)

from scipy.constants import h, e

from Circuit import Circuit
from quantization import quantize
from solver import drive_params, calculate_rabi_period
from DriveParams import DriveParams
from plotting import plot_all
from LivePlotter import LivePlotter
from config import *


def main():
    # -------------------------------------------------------------------------
    #   Hyper-Parameters
    # -------------------------------------------------------------------------

    fC   = 250e6    # Charging energy frequency EC/h [Hz]
    n_cut = 20

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
    CS    = C_sum
    L     = 80e-9

    # -------------------------------------------------------------------------
    #   Circuit Graph Representation
    # -------------------------------------------------------------------------

    graph_rep = {
        'nodes': ['a'],
        'capacitors': [('a', 'gnd', CS)],
        'inductors': [('a', 'gnd', L)],
        'josephson_elements': [],
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
    print("Quantization Complete.")
    print("Fock Operator (n_hat):")
    print(quant.n_hat)
    print("Hamiltonian in Fock Basis (H_hat) [GHz]:")
    print(quant.H_hat / (h * 1e9))
    print("Eigenvalues [GHz]:")
    print(quant.energies / (h * 1e9))
    print("phi Operator in Energy Basis:")
    print(quant.phi_hat_energy)
    print(f"# Energy Levels: {quant.states.shape[0]}")

    # -------------------------------------------------------------------------
    #   Harmonic Oscillator Parameter Summary
    # -------------------------------------------------------------------------

    f0, f1, f2 = quant.energies[0:3] / h / 1e9
    alpha = (f2 - f1) - (f1 - f0)

    print("Harmonic Oscillator parameters:")
    print(f"  EC/h          = {EC / (h * 1e9):.3f} GHz")
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
    if rabi_period:
        print(f"Rabi Period: {rabi_period * 1e9:.2f} ns")


if __name__ == "__main__":
    main()
