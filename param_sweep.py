import numpy as np
np.set_printoptions(precision=8, linewidth=120, suppress=True)
from Circuit import *
from scipy.constants import h, e
from config import *
import matplotlib.pyplot as plt

def parameter_sweep(circuit: Circuit, crank_nic_params: list, sweep_idx: int, min_param: float, max_param: float, resolution: int):
    params = np.linspace(min_param, max_param, resolution)
    
    fidelities = []
    
    for param_val in params:
        crank_nic_params[sweep_idx] = param_val
        circuit._calculate_rabi_period(*crank_nic_params)

        if circuit.rabi_period is None:
            fidelities.append(0)
            continue

        d = 2
        circuit._build_unitary(
            dim_sub=crank_nic_params[0],
            d=d,
            detuning=crank_nic_params[2],
            amplitude_scale=crank_nic_params[4],
            sigma=crank_nic_params[5],
            lambda_drag=crank_nic_params[6],
            steps_per_period=crank_nic_params[7])

        fidelity = circuit._calculate_fidelity(d=d)
        fidelities.append(fidelity)
        
    return params, fidelities
    
    
def main():
    # Cooper Pair Box / Transmon-like circuit
    # Just a Josephson junction with shunt capacitance (no linear inductor)
    
    ################################# HYPER-PARAMETERS #################################
    
    # Define energies
    fC    = 250e6  # Charging energy frequency EC/h [Hz]
    EJ_EC = 50     # EJ/EC ratio (typical transmon)
    
    # Charge Basis Cut-Off
    n_cut             = 20
    
    # Crank-Nicolson Parameters
    dim_sub           = 6
    init_state        = 0
    detuning          = OPTIMAL_DETUNING
    N_pulses          = 1000 
    amplitude_scale   = OPTIMAL_AMPLITUDE_SCALE
    sigma             = OPTIMAL_SIGMA 
    lambda_drag       = OPTIMAL_LAMBDA_DRAG
    steps_per_period  = 200
    
    crank_nic_params = [dim_sub, init_state, detuning, N_pulses, amplitude_scale, sigma, lambda_drag, steps_per_period]
    
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
    
    
    params, fidelities = parameter_sweep(
        circuit=circuit, 
        crank_nic_params=crank_nic_params, 
        sweep_idx=6, # lambda DRAG
        min_param=0.0, 
        max_param=0.02,
        resolution=21
    )
    
    print(f"Best Param Value: {params[np.argmax(np.array(fidelities))]}")
         
    plt.figure(figsize=(8, 4))
    plt.plot(params, fidelities, linewidth=2)
    plt.xlabel('Parameter Value')
    plt.ylabel('Fidelity')
    plt.title('Fidelity vs Parameter Value')
    plt.grid(True)
    plt.show()
    
    
    
if __name__=="__main__":
    main()