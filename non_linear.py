###############################################################################
#                                                                             #
#   non_linear.py                                                             #
#                                                                             #
#   Simulates a Cooper Pair Box / Transmon-like superconducting circuit       #
#   consisting of a single Josephson junction with shunt capacitance.         #
#   Performs quantization, diagonalization, Rabi period calculation,          #
#   unitary construction, and gate fidelity evaluation.                       #
#                                                                             #
###############################################################################

import numpy as np
np.set_printoptions(precision=8, linewidth=120, suppress=True)
from Circuit import Circuit
from plotting import plot_all
from LivePlotter import LivePlotter
from scipy.constants import h, e
from config import *

def main():
    # =========================================================================
    #   Cooper Pair Box / Transmon-like circuit
    #   Just a Josephson junction with shunt capacitance (no linear inductor)
    # =========================================================================

    # -------------------------------------------------------------------------
    #   Hyper-Parameters
    # -------------------------------------------------------------------------

    # --- Energy scales ---
    fC    = 250e6  # Charging energy frequency EC/h [Hz]
    EJ_EC = 50     # EJ/EC ratio (typical transmon)
    
    # --- Charge-basis cut-Off
    n_cut             = 20

    # --- Crank-Nicolson time-evolution parameters ---
    init_state        = 0
    dim_sub           = 6
    detuning          = OPTIMAL_DETUNING
    N_pulses          = 2000
    amplitude_scale   = OPTIMAL_AMPLITUDE_SCALE
    sigma             = OPTIMAL_SIGMA
    lambda_drag       = OPTIMAL_LAMBDA_DRAG
    steps_per_period  = 200
    
    crank_nic_params = [dim_sub, init_state, detuning, N_pulses, amplitude_scale, sigma, lambda_drag, steps_per_period]

    # -------------------------------------------------------------------------
    #   Derived Physical Constants
    # -------------------------------------------------------------------------

    EC        = h * fC            # Charging energy              [J]
    C_sum     = (e**2) / (2 * EC) # Total capacitance            [F]

    CJ        = 0.05 * C_sum      # 5%  of capacitance from JJ   [F]
    CS        = 0.95 * C_sum      # 95% of capacitance from shunt [F]

    EJ        = EJ_EC * EC        # Josephson energy              [J]

    # -------------------------------------------------------------------------
    #   Circuit Graph Representation
    # -------------------------------------------------------------------------

    graph_rep = {
        'nodes': ['a'],
        'capacitors': [
            ('a', 'gnd', CS), # Shunt Capacitor
            ('a', 'gnd', CJ)  # Josephson Junction Capacitor
            ],
        'inductors': [],
        'josephson_elements': [
            ('a', 'gnd', EJ), # Josephson Element
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
    print(f"Inverse Inductance Matrix [1/nH]: ")
    print(circuit.inv_inductance_matrix / 1e9)

    # -------------------------------------------------------------------------
    #   Step 1: Quantize — build charge & Hamiltonian operators (charge basis)
    # -------------------------------------------------------------------------

    circuit._quantize(n_cut=n_cut)
    print("Quantization Complete...")
    print("Charge Operator (n_hat) [dimensionless]: ")
    print(circuit.n_hat)
    print("Hamiltonian Operator in Charge Basis (H_hat) [GHz]: ")
    print(circuit.H_hat / (h * 1e9))

    # -------------------------------------------------------------------------
    #   Step 2: Diagonalize — obtain energy eigenvalues and eigenstates
    # -------------------------------------------------------------------------

    circuit._diagonalize()
    print("Diagonalization Complete...")
    print("Eigenvalues of Hamiltonian Operator in Charge Basis [GHz]: ")
    print(circuit.energies / (h * 1e9))
    print("Eigenvectors of Hamiltonian Operator in Charge Basis [dimensionless]: ")
    print("(Columns Are Eigenvectors)")
    print(circuit.states)

    # -------------------------------------------------------------------------
    #   Step 3: Change basis — express charge operator in the energy basis
    # -------------------------------------------------------------------------

    circuit._change_basis()
    print("Changed Charge Operator Basis...")
    print("Charge Operator in Energy Basis [dimensionless]: ")
    print(circuit.n_hat_energy)

    print(f"# Energy Levels: {circuit.states.shape[0]}")

    # -------------------------------------------------------------------------
    #   Transmon Parameter Summary
    # -------------------------------------------------------------------------

    f0, f1, f2 = circuit.energies[0:3] / h / 1e9 # [Ghz]
    alpha = (f2 - f1) - (f1 - f0)

    print("Transmon parameters:")
    print(f"  EC/h  = {(EC / (h * 1e9)):.3f} GHz")
    print(f"  EJ/EC = {(EJ / EC):.1f}")
    print(f"  Anharmonicity = {alpha:.3f} GHz")

    # -------------------------------------------------------------------------
    #   Step 4: Calculate Rabi period & plot energy landscape
    # -------------------------------------------------------------------------

    _, f_drive, _ = circuit._drive_params(detuning)
    
    if LIVE_VISUALIZATION == True:
        live_plotter = LivePlotter(circuit=circuit, 
                                   f_drive=f_drive, 
                                   dim_sub=dim_sub, 
                                   n_cut=n_cut,
                                   min_flux=-np.pi, 
                                   max_flux=np.pi,
                                   update_interval=500
                                   )
        circuit._calculate_rabi_period(*crank_nic_params, callback=live_plotter.update)
        live_plotter.finalize()
    else:
        circuit._calculate_rabi_period(*crank_nic_params, callback=None)

    rabi_period = circuit.rabi_period

    plot_all(circuit=circuit, n_cut=n_cut, min_flux=-np.pi, max_flux=np.pi, num_phases=1000, detuning=detuning)

    print(f"Rabi Period: {rabi_period * 1e9:.2f} ns")

    # -------------------------------------------------------------------------
    #   Step 5: Build unitary gate & evaluate fidelity
    # -------------------------------------------------------------------------

    # Dimension of the computational subspace (single qubit -> d=2)
    d=2

    circuit._build_unitary(
        d=d,
        dim_sub=dim_sub,
        detuning=detuning,
        amplitude_scale=amplitude_scale,
        sigma=sigma,
        lambda_drag=lambda_drag,
        steps_per_period=steps_per_period)

    fidelity = circuit._calculate_fidelity(d=d)

    print("Actual Unitary Operator: ")
    print(circuit.U)

    print("Top-Left d x d of self.U: ")
    print(circuit.U[:2, :2])

    print(f"Gate Fidelity: {fidelity}")


if __name__=="__main__":
    main()
