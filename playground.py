import numpy as np
from Circuit import Circuit, plot_charge_distribution, plot_potential_energy, plot_pulse_sequence, plot_rabi_oscillations
from scipy.constants import h, e
    
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
    delta_f           = -1e6    
    N_pulses          = 1000    
    sigma             = 15e-12  
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
    print(f"Capacitance Matrix: {circuit.capacitance_matrix}")
    print(f"Inverse Inductance Matrix: {circuit.inv_inductance_matrix}")
    
    # Get charge operator and Hamiltonian operator in the charge basis
    circuit._quantize(n_cut=n_cut)
    # Diagonalize Hamiltonian operator to get energies and states
    circuit._diagonalize()
    # Crate new charge operator in energy basis
    circuit._change_basis()
    
    print(f"# Energy Levels: {circuit.states.shape[0]}")
    
    f0, f1, f2 = circuit.energies[0:3] / h / 1e9 # [Ghz]
    alpha = (f2 - f1) - (f1 - f0)
    
    print("Transmon parameters:")
    print(f"  EC/h  = {(EC / h):.3f} GHz")
    print(f"  EJ/EC = {(EJ / EC):.1f}")
    print(f"  Anharmonicity = {alpha:.3f} GHz")  
    
    # Drive frequency and SFQ pulse parameters
    f01_Hz   = (f1 - f0) * 1e9       # |0> --> |1>  [Hz]
    f12_Hz   = (f2 - f1) * 1e9       # |1> --> |2>  [Hz]
    f_drive  = f01_Hz + delta_f      # drive repetition rate resonant with 1-2 + detuning
    T_drive  = 1 / f_drive           # pulse repetition period [s]
    
    circuit.crank_nicolson(
        dim_sub=dim_sub, 
        T_drive=T_drive, 
        N_pulses=N_pulses, 
        sigma=sigma, 
        steps_per_period=steps_per_period
        )  
        
    plot_charge_distribution(n_cut=n_cut, states=circuit.states)
    
    plot_potential_energy(circuit=circuit, min_flux=-np.pi, max_flux=np.pi)
    
    plot_pulse_sequence(t_vec=circuit.t_vec, At_vec=circuit.At_vec)
    
    plot_rabi_oscillations(T_drive=T_drive, t_vec=circuit.t_vec, P_0=circuit.P_0, P_1=circuit.P_1, P_2=circuit.P_2)

    
    
if __name__=="__main__":
    main()