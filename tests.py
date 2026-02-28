import numpy as np
from scipy.constants import e, hbar

from Circuit import Circuit
from constants import PHI_0


def test_1():
    EJ = 20e-23       # ~20 GHz Josephson energy
    CJ = 5e-15        # 5 fF junction capacitance
    CS = 100e-15      # 100 fF shunt capacitance (transmon regime)

    graph_rep = {
        'nodes': ['a'],
        'capacitors': [('a', 'gnd', CS), ('a', 'gnd', CJ)],
        'inductors': [],
        'josephson_elements': [('a', 'gnd', EJ)],
        'external_flux': {}
    }

    circuit = Circuit(graph_rep)

    assert np.isclose(circuit.capacitance_matrix[0][0], CS + CJ), "Capacitance Matrix Error"
    assert np.isclose(circuit.inv_inductance_matrix[0][0], 0),    "Inductance Matrix Error"

    phi_test = np.array([0.1 * PHI_0])
    q_test   = np.array([1e-19])

    assert np.isclose(
        circuit.get_kinetic_energy(q_test),
        0.5 * q_test[0] * (1 / (CS + CJ)) * q_test[0]
    ), "Kinetic Energy Error"

    assert np.isclose(
        circuit.get_potential_energy(phi_test),
        -EJ * np.cos(-phi_test[0] / PHI_0)
    ), "Potential Energy Error"


def main():
    test_1()
    print("test_1 passed")


if __name__ == "__main__":
    main()
