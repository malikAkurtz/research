from circuit_hamiltonian import Circuit, PHI_0
import numpy as np
from scipy.constants import e, hbar

def test_1():
    EJ = 20e-23      # ~20 GHz Josephson energy (h * 20 GHz ≈ 1.3e-23 J)
    CJ = 5e-15       # 5 fF junction capacitance
    CS = 100e-15     # 100 fF shunt capacitance (transmon regime)
    
    graph_rep = {
        'nodes': ['a'],
        'capacitors': [('a', 'gnd', CS)],
        'inductors': [],  # Pure Josephson - no linear inductors
        'josephson_junctions': [
            ('a', 'gnd', EJ, CJ),
        ],
        'external_flux': {}
    }
    
    circuit = Circuit(graph_rep)
    
    
    assert np.isclose(circuit.capacitance_matrix[0][0], (CS + CJ)), "Capacitance Matrix Error"
    assert np.isclose(circuit.inv_inductance_matrix[0][0], 0), "Inductance Matrix Error"
    
    phi_test = np.zeros(circuit.N, dtype=float).reshape(-1)  # Force 1D
    phi_test[0] = 0.1 * PHI_0 
    q_test = (np.ones(circuit.N) * 1e-19).reshape(-1)  # Force 1D
    
    assert np.isclose(circuit.get_kinetic_energy(q_test), (1/2)*q_test[0]*(1/(CS + CJ))*q_test[0]), "Kinetic Energy Error"
    assert np.isclose(circuit.get_potential_energy(phi_test), -EJ*np.cos(-phi_test[0]/PHI_0)), "Potential Energy Error"
    

def main():
    test_1()
    print("test_1 passed")
    
if __name__=="__main__":
    main()