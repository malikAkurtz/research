import numpy as np
from Circuit import Circuit
from scipy.constants import h, e
    
def main():
    
    ################################# HYPER-PARAMETERS #################################
    
    # Define energies
    fC    = 250e6  # Charging energy frequency EC/h [Hz]
    
    # 4. Simulation Parameters
    n_cut = 20
    delta_f = -1e6    
    
    ################################# HYPER-PARAMETERS #################################
    
    EC = h * fC
    C_sum = (e**2) / (2 * EC) 
    
    CS = C_sum
    
    L = 80e-9
    graph_rep = {
        'nodes': ['a'],
        'capacitors': [('a', 'gnd', CS)],
        'inductors': [('a', 'gnd', L)],
        'josephson_junctions': [],
        'external_flux': {}
    }
    
    circuit = Circuit(graph_rep=graph_rep, n_cut=n_cut)
    
    for node in circuit.active_nodes:
        print(f"  Active node: {node.label}")
    for node in circuit.passive_nodes:
        print(f"  Passive node: {node.label}")
    
    print(circuit)
    print(f"Capacitance Matrix: {circuit.capacitance_matrix}")
    C_sum = circuit.capacitance_matrix[0][0]
    EC = e**2 / (2*C_sum)
    print(f"Inverse Inductance Matrix: {circuit.inv_inductance_matrix}")
    
    print(f"# Energy Levels: {circuit.states.shape[0]}")
    
    f0, f1, f2 = circuit.energies[0:3] / h / 1e9 # Convert to GHz
    alpha = (f2 - f1) - (f1 - f0)
    
    print("Circuit parameters:")
    print(f"  EC/h  = {(EC / h):.3f} GHz")
    print(f"  Anharmonicity = {alpha:.3f} GHz")  
         
    circuit.plot_charge_distribution()
    
    circuit.plot_Josephson_potential(min_flux=-np.pi, max_flux=np.pi)
    

    
    
if __name__=="__main__":
    main()