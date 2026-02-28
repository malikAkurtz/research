import numpy as np
from Circuit import *
from scipy.constants import h, e
from config import *
from LivePlotter import *
from plotting import *
    
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
    
    EC = h * fC
    C_sum = (e**2) / (2 * EC) 
    
    CS = C_sum
    
    L = 80e-9
    
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
    print(f"Inverse Inductance Matrix [1/nH]: ")
    print(circuit.inv_inductance_matrix / 1e9)

    # -------------------------------------------------------------------------
    #   Step 1: Quantize — build Fock & Hamiltonian operators (Fock basis)
    # -------------------------------------------------------------------------

    circuit._quantize(n_cut=n_cut)
    print("Quantization Complete...")
    print("Fock Operator (n_hat) [dimensionless]: ")
    print(circuit.n_hat)
    print("Hamiltonian Operator in Fock Basis (H_hat) [GHz]: ")
    print(circuit.H_hat / (h * 1e9))

    # -------------------------------------------------------------------------
    #   Step 2: Diagonalize — obtain energy eigenvalues and eigenstates
    # -------------------------------------------------------------------------

    circuit._diagonalize()
    print("Diagonalization Complete...")
    print("Eigenvalues of Hamiltonian Operator in Fock Basis [GHz]: ")
    print(circuit.energies / (h * 1e9))
    print("Eigenvectors of Hamiltonian Operator in Fock Basis [dimensionless]: ")
    print("(Columns Are Eigenvectors)")
    print(circuit.states)

    # -------------------------------------------------------------------------
    #   Step 3: Change basis — express PHI operator in the energy basis
    # -------------------------------------------------------------------------

    circuit._change_basis()
    print("Changed Charge Operator Basis...")
    print("Charge Operator in Energy Basis [dimensionless]: ")
    print(circuit.PHI_hat_energy)

    print(f"# Energy Levels: {circuit.states.shape[0]}")

    # -------------------------------------------------------------------------
    #   Harmonic Oscillator Parameter Summary
    # -------------------------------------------------------------------------

    f0, f1, f2 = circuit.energies[0:3] / h / 1e9 # [Ghz]
    alpha = (f2 - f1) - (f1 - f0)

    print("Transmon parameters:")
    print(f"  EC/h  = {(EC / (h * 1e9)):.3f} GHz")
    print(f"  Anharmonicity = {alpha:.3f} GHz")

    # -------------------------------------------------------------------------
    #   Step 4: Calculate Rabi period & plot energy landscape
    # -------------------------------------------------------------------------

    _, f_drive, _ = circuit._drive_params(detuning)
    
    if LIVE_VISUALIZATION == True:
        live_plotter = LivePlotter(
                                   basis = circuit.basis,
                                   f_drive=f_drive, 
                                   dim_sub=dim_sub, 
                                   n_cut=n_cut,
                                   min_flux=-np.pi, 
                                   max_flux=np.pi,
                                   update_interval=5
                                   )
        circuit._calculate_rabi_period(*crank_nic_params, callback=live_plotter.update)
        live_plotter.finalize()
    else:
        circuit._calculate_rabi_period(*crank_nic_params, callback=None)

    rabi_period = circuit.rabi_period

    plot_all(circuit=circuit, n_cut=n_cut, min_flux=-np.pi, max_flux=np.pi, num_phases=1000, detuning=detuning)

    print(f"Rabi Period: {rabi_period * 1e9:.2f} ns")



if __name__=="__main__":
    main()
