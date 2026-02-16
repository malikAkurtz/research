import numpy as np
np.set_printoptions(precision=8, linewidth=120, suppress=True)
from Circuit import *
from scipy.constants import h, e
from config import *

def parameter_sweep(circuit: Circuit, crank_nic_params: np.ndarray, sweep_idx: int, min_param: float, max_param: float, resolution: int):
    params = np.linspace(min_param, max_param, resolution)
    
    max_P1s = []
    
    for param in params:
        crank_nic_params[sweep_idx] = param
        circuit.crank_nicolson(*crank_nic_params)

        max_P1s.append(max(circuit.P_1))
        
    return params, max_P1s
    
    
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
    
    circuit._quantize(n_cut=n_cut)
    circuit._diagonalize()
    circuit._change_basis()
    
    crank_nic_params = [dim_sub, detuning, N_pulses, amplitude_scale, sigma, steps_per_period]
    
    params, max_P1s = parameter_sweep(
        circuit=circuit, 
        crank_nic_params=crank_nic_params, 
        sweep_idx=4, # Sigma
        min_param=5e-12, 
        max_param=25e-12,
        resolution=21
    )
    
    print(f"Best Param Value: {params[np.argmax(np.array(max_P1s))]}")
         
    plt.figure(figsize=(8, 4))
    plt.plot(params, max_P1s, linewidth=2)
    plt.xlabel('Parameter Value')
    plt.ylabel('P(Measure|1>)')
    plt.title('P1 vs Parameter Value')
    plt.grid(True)
    plt.show()
    
    
    
if __name__=="__main__":
    main()