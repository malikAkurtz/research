import numpy as np
from Circuit import Circuit, plot_potential_energy
from scipy.constants import h, e
    
def main():
    
    ################################# HYPER-PARAMETERS #################################
    
    # Define energies
    fC    = 250e6  # Charging energy frequency EC/h [Hz]
    
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
    
    circuit = Circuit(graph_rep=graph_rep)
    
    for node in circuit.active_nodes:
        print(f"  Active node: {node.label}")
    for node in circuit.passive_nodes:
        print(f"  Passive node: {node.label}")
    
    print(circuit)
    print(f"Capacitance Matrix: {circuit.capacitance_matrix}")
    print(f"Inverse Inductance Matrix: {circuit.inv_inductance_matrix}")
    
    omega = np.sqrt(1/L*C_sum)
    
    # Calculate the first 100 energies
    circuit.energies = np.array([h*(omega)*(n + 0.5) for n in range(10)])
    print(circuit.energies)
        
    plot_potential_energy(circuit=circuit, min_flux=-100000, max_flux=100000)
    

    
    
if __name__=="__main__":
    main()