import numpy as np
np.set_printoptions(precision=8, linewidth=120, suppress=True)
from Circuit import *
from scipy.constants import h, e
from config import OPTIMAL_DETUNING, OPTIMAL_AMPLITUDE_SCALE, OPTIMAL_SIGMA
    
def main():
    # Cooper Pair Box / Transmon-like circuit
    # Just a Josephson junction with shunt capacitance (no linear inductor)
    
    ################################# HYPER-PARAMETERS #################################
    
    # Define energies
    fC    = 250e6  # Charging energy frequency EC/h [Hz]
    EJ_EC = 50     # EJ/EC ratio (typical transmon)
    
    # Crank-Nicolson Parameters
    n_cut             = 20
    dim_sub           = 6
    detuning          = OPTIMAL_DETUNING
    N_pulses          = 1000
    amplitude_scale   = OPTIMAL_AMPLITUDE_SCALE
    sigma             = OPTIMAL_SIGMA
    steps_per_period  = 200
    
    ################################# HYPER-PARAMETERS #################################
    
    EC        = h * fC            # [J]
    C_sum     = (e**2) / (2 * EC) # [F]
    
    CJ        = 0.05 * C_sum # 5% of capacitance from Josephson Junction
    CS        = 0.95 * C_sum # 95% of capacitiance from shunt capacitor
    
    EJ        = EJ_EC * EC        # [J]
    
    graph_rep = {
        'nodes': ['a'],
        'capacitors': [('a', 'gnd', CS)],
        'inductors': [],
        'josephson_junctions': [
            ('a', 'gnd', EJ, CJ),
        ],
        'external_flux': {}
    }
    
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
    
    # Get charge operator and Hamiltonian operator in the charge basis
    circuit._quantize(n_cut=n_cut)
    print("Quantization Complete...")
    print("Charge Operator (n_hat) [dimensionless]: ")
    print(circuit.n_hat)
    print("Hamiltonian Operator in Charge Basis (H_hat) [GHz]: ")
    print(circuit.H_hat / (h * 1e9))
    
    # Diagonalize Hamiltonian operator to get energies and states
    circuit._diagonalize()
    print("Diagonalization Complete...")
    print("Eigenvalues of Hamiltonian Operator in Charge Basis [GHz]: ")
    print(circuit.energies / (h * 1e9))
    print("Eigenvectors of Hamiltonian Operator in Charge Basis [dimensionless]: ")
    print("(Columns Are Eigenvectors)")
    print(circuit.states)
    
    # Crate new charge operator in energy basis
    circuit._change_basis()
    print("Changed Charge Operator Basis...")
    print("Charge Operator in Energy Basis [dimensionless]: ")
    print(circuit.n_hat_energy)
    
    print(f"# Energy Levels: {circuit.states.shape[0]}")
    
    f0, f1, f2 = circuit.energies[0:3] / h / 1e9 # [Ghz]
    alpha = (f2 - f1) - (f1 - f0)
    
    print("Transmon parameters:")
    print(f"  EC/h  = {(EC / (h * 1e9)):.3f} GHz")
    print(f"  EJ/EC = {(EJ / EC):.1f}")
    print(f"  Anharmonicity = {alpha:.3f} GHz")  
    
    circuit._calculate_rabi_period(dim_sub, detuning, N_pulses, amplitude_scale, sigma, steps_per_period)
    
    rabi_period = circuit.rabi_period
    
    print(f"Rabi Period: {rabi_period * 1e9:.2f} ns")
    
    # plot_all(circuit=circuit, n_cut=n_cut, min_flux=-np.pi, max_flux=np.pi, num_phases=1000, detuning=detuning)
    
    # dimension of subspace of interest (single qubit)
    d=2
    
    circuit._build_unitary(
        dim_sub=dim_sub, 
        d=d,
        detuning=detuning, 
        amplitude_scale=amplitude_scale,
        sigma=sigma, 
        steps_per_period=steps_per_period)
    
    fidelity = circuit._calculate_fidelity(d=d)
    
    print(f"Gate Fidelity: {fidelity}")

    
    
if __name__=="__main__":
    main()