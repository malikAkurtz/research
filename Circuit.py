import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, hbar, e

from Elements import Node, Capacitor, Inductor, JosephsonJunction, Graph

PHI_0 = hbar / (2 * e)

######################################## CIRCUIT CLASS ########################################
class Circuit:
    """
    Circuit class to build and store relevant data for a circuit
    using the method of nodes.
    """
    def __init__(self, graph_rep: dict):
        self.labels                                           = graph_rep["nodes"] # ["a", "b", "c", etc.]
        self.circuit_graph                                    = Circuit._build_master_graph(graph_rep) # collection of edges,vertices
        self.capacitive_sub_graph, self.inductive_sub_graph   = self._build_sub_graphs() # capactive edges,vertices + inductive edges,vertices
        self.active_nodes, self.passive_nodes                 = self._partition_nodes()
        self.N                                                = len(self.active_nodes)
        self.P                                                = self.N + len(self.passive_nodes) + 1 # for ground
        self.capacitance_matrix, self.inv_inductance_matrix   = self._build_matrices()
        self.inv_capacitance_matrix                           = np.linalg.inv(self.capacitance_matrix)
        self.offset_dict                                      = graph_rep["external_flux"]
        self.omega_squared                                    = self._build_omega_squared()
        self.normal_modes_squared, self.normal_vecs_squared   = np.linalg.eig(self.omega_squared)
        self.josephson_junctions                              = graph_rep["josephson_junctions"]
        
        self.n_hat, self.H_hat                                = None, None
        self.energies, self.states                            = None, None
        self.n_hat_energy                                     = None
        self.t_vec, self.At_vec, self.P_0, self.P_1, self.P_2 = None, None, None, None, None
        
    @staticmethod
    def _build_master_graph(graph_rep: dict) -> Graph:
        gnd = Node(label="gnd")
        
        nodes    = {"gnd" : gnd}
        branches = []
        
        # create nodes
        for i in range(len(graph_rep["nodes"])):
            # save the node label, e.g. "a"
            node_label = graph_rep["nodes"][i]
            # create a new Node object and index it with key "node_label"
            nodes[node_label] = Node(label=node_label)
    
        # create and assign capacitive branches
        for i in range(len(graph_rep['capacitors'])):
            # save the capacitor tuple
            entry = graph_rep['capacitors'][i]
            node1 = nodes[entry[0]]
            node2 = nodes[entry[1]]
            value = entry[2]
            
            # create the branch object with nodes as arguments
            capacitor = Capacitor(value=value, nodes=[node1, node2])
            
            branches.append(capacitor)
            node1.branches.append(capacitor)
            node2.branches.append(capacitor)
            
        # create and assign inductive branches
        for i in range(len(graph_rep['inductors'])):
            # save the inductor tuple
            entry = graph_rep['inductors'][i]
            node1 = nodes[entry[0]]
            node2 = nodes[entry[1]]
            value = entry[2]
            
            # create the branch object with nodes as arguments
            inductor = Inductor(value=value, nodes=[node1, node2])
            
            branches.append(inductor)
            node1.branches.append(inductor)
            node2.branches.append(inductor)
            
        # create and assign Josephson Junction capacitive branches
        for i in range(len(graph_rep["josephson_junctions"])):
            # save the jj tuple
            entry = graph_rep["josephson_junctions"][i]
            node1 = nodes[entry[0]]
            node2 = nodes[entry[1]]
            EJ = entry[2]
            CJ = entry[3]
            
            josephson_junction = JosephsonJunction(EJ=EJ, CJ=CJ, nodes=[node1, node2])
            
            branches.append(josephson_junction)
            node1.branches.append(josephson_junction)
            node2.branches.append(josephson_junction)
            
        # break symmetry
        for node in nodes.values():
            if node.label == "gnd":
                continue
            has_capacitive_branch = False
            for branch in node.branches:
                if (type(branch) is Capacitor) or (type(branch) is JosephsonJunction):
                    has_capacitive_branch = True
                    break
            if not has_capacitive_branch:
                capacitor = Capacitor(value=1e-20, nodes=[node, nodes["gnd"]])
                
                branches.append(capacitor)
                node.branches.append(capacitor)
                nodes["gnd"].branches.append(capacitor)
                        
        return Graph(nodes, branches)
            
    def _build_sub_graphs(self) -> dict:
        nodes    = self.circuit_graph.vertices
        branches = self.circuit_graph.edges
        
        capacitive_branches = []
        inductive_branches  = []
        
        # filter  branches
        for branch in branches:
            if type(branch) in [Capacitor, JosephsonJunction]:
                capacitive_branches.append(branch)
            if type(branch) in [Inductor, JosephsonJunction]:
                inductive_branches.append(branch)
        
        return Graph(nodes, capacitive_branches), Graph(nodes, inductive_branches)
    
    def _partition_nodes(self):
        active_nodes    = []
        passive_nodes   = []
        
        # assuming symmetry already broken, only have to check inductive degree
        for node_label, node in self.circuit_graph.vertices.items():
            if node_label == "gnd":
                continue
            capacitive_degree, inductive_degree = node._get_degree()
            if (inductive_degree > 0):
                active_nodes.append(node)
            else:
                passive_nodes.append(node)
                
        return active_nodes, passive_nodes
    
    def _build_matrices(self):
        capacitance_matrix      = np.zeros((self.P, self.P))
        inv_inductance_matrix   = np.zeros((self.P, self.P))
        
        for branch in self.circuit_graph.edges:
            # e.g. "a", "b", etc.
            node1_label = branch.nodes[0].label
            node2_label = branch.nodes[1].label
            
            # get index of label from nodes = ["a", "b", "c", ...,]
            if node1_label == "gnd":
                j = 0
            else:
                j = self.labels.index(node1_label) + 1
            
            if node2_label == "gnd":
                k = 0
            else:
                k = self.labels.index(node2_label) + 1
                        
            if (type(branch) is Capacitor) or (type(branch) is JosephsonJunction):
                capacitance_matrix[j][k] += -branch.C
                capacitance_matrix[k][j] += -branch.C
            else:
                inv_inductance_matrix[j][k] += -1 / branch.L
                inv_inductance_matrix[k][j] += -1 / branch.L
                
        for j in range(self.P):
            capacitance_matrix[j][j]    = -1 * np.sum(capacitance_matrix[j]).item()
            inv_inductance_matrix[j][j] = -1 * np.sum(inv_inductance_matrix[j]).item()
            
        capacitance_matrix = np.delete(capacitance_matrix, 0, axis=0)
        capacitance_matrix = np.delete(capacitance_matrix, 0, axis=1)
        
        inv_inductance_matrix = np.delete(inv_inductance_matrix, 0, axis=0)
        inv_inductance_matrix = np.delete(inv_inductance_matrix, 0, axis=1)
    
        return capacitance_matrix, inv_inductance_matrix
    
    def _get_node_flux_dot(self, node_charge):
        return self.inv_capacitance_matrix @ node_charge
    
    def get_kinetic_energy(self, node_charge):
        node_flux_dot = self._get_node_flux_dot(node_charge)
        return (node_flux_dot.T @ self.capacitance_matrix @ node_flux_dot) / 2
    
    def get_potential_energy(self, node_flux: np.ndarray):
        if self.inv_inductance_matrix.size > 0 and np.any(self.inv_inductance_matrix != 0):
            term_1 = (node_flux.T @ self.inv_inductance_matrix @ node_flux) / 2
        else:
            term_1 = 0
            
        term_2 = 0
        # loop through offsets and add their terms
        for (node1_label, node2_label), offset in self.offset_dict.items():
                j = self.labels.index(node1_label)
                k = self.labels.index(node2_label)
                
                inductance = -1 / self.inv_inductance_matrix[j][k]
                term_2 += ((node_flux[j] - node_flux[k]) * offset) / inductance
        
        linear_term = term_1 + term_2
        
        non_linear_term = 0
        phi0 = hbar / (2 * e)
        
        for jj in self.josephson_junctions:
            node1_label = jj[0]
            node2_label = jj[1]
            EJ          = jj[2]
            
            node1_flux = node_flux[self.labels.index(node1_label)] if node1_label != "gnd" else 0
            node2_flux = node_flux[self.labels.index(node2_label)] if node2_label != "gnd" else 0
            
            branch_flux = node1_flux - node2_flux
            
            phi = branch_flux / phi0
            
            non_linear_term += -EJ * np.cos(phi)
                
        return linear_term + non_linear_term
    
    def get_lagrangian(self, node_flux, node_charge):
        return self.get_kinetic_energy(node_charge) - self.get_potential_energy(node_flux)
    
    def _build_omega_squared(self):
        # Only compute normal modes if we have linear inductors
        if np.any(self.inv_inductance_matrix != 0):
            return self.inv_capacitance_matrix @ self.inv_inductance_matrix
        else:
            # Return zeros for pure nonlinear circuits
            return np.zeros((self.N, self.N))
    
    def get_hamiltonian(self, node_flux, node_charge):
        return 0.5 * (node_charge.T @ self.inv_capacitance_matrix @ node_charge) + self.get_potential_energy(node_flux)
    
    @staticmethod
    def get_LJ(EJ: float):
        return PHI_0**2 / EJ
    
    # Currently only works for a circuit with a single node and single Josephson Junction
    def _quantize(self, n_cut: int):
        C_sum = self.capacitance_matrix[0][0]
        
        if self.josephson_junctions:
            EJ = self.josephson_junctions[0][2]
        
        # Cost to add one Cooper pair of charge
        EC = e**2 / (2*C_sum)

        # Define Hilbert space
        # e.g. [-20, -19, ..., 0, ..., 19, 20] for n_cut = 20
        # each n represents n Cooper pairs on the island
        n_vals = np.array([i for i in range(-n_cut, n_cut+1)])
        dim = len(n_vals)
        n_hat = np.diag(n_vals)
                
        # Build Hamiltonian Operator
        H = (4 * EC * (n_hat @ n_hat)) - 0.5 * EJ * (np.diag(np.ones(dim-1), 1) + np.diag(np.ones(dim-1), -1))
        
        self.n_hat = n_hat
        self.H_hat = H
    
    def _diagonalize(self):
        self.energies, self.states = np.linalg.eigh(self.H_hat)
        
    def _change_basis(self):
        self.n_hat_energy = np.conjugate(self.states).T @ self.n_hat @ self.states
    
    def crank_nicolson(self, dim_sub: int, T_drive: float, N_pulses: int, sigma: float, steps_per_period: int):
        # Only grab the dim_sub lowest eigenvalues
        truncated_energies = self.energies[:dim_sub]
        
        # Truncated independent Hamiltonian
        H_0 = np.diag(truncated_energies)
        
        # Truncated charge operator in energy basis
        n_op = self.n_hat_energy[:dim_sub, :dim_sub]
        
        # Total simulation time (continuous)
        T = N_pulses * T_drive
        
        # delta t
        dt = T_drive / steps_per_period
        
        # Number of delta ts
        N_t = round(T / dt)
        
        # Time evolution vector
        t_vec = np.array([i for i in range(N_t)]).T * dt
        
        # Pulse amplitude in Joules 
        A_0 = 0.01 * (truncated_energies[1]-truncated_energies[0])
        
        # Precompute pulse centers
        pulse_centers = np.array([i for i in range(N_pulses)]) * T_drive;
        
        # Initial state: ground state |0> in eigenbasis (energy basis)
        psi = np.zeros(dim_sub);
        psi[0] = 1.0;
        
        # Pre-allocate arrays for results
        At_vec = np.zeros(N_t)
        P_0 = np.zeros(N_t)
        P_1 = np.zeros(N_t)
        P_2 = np.zeros(N_t)
        
        # Identity matrix
        I = np.eye(dim_sub)
        
        # Crank-Nicolson time evolution
        for i in range(N_t):
            # Actual time at this time-step
            t = t_vec[i]
            t_mid = t + (dt / 2)
            
            # Need to calculate H_mid
            
            # SFQ pulse train envelope: sum of Gaussians centered at pulse_centers
            # Only sum nearby pulses (within ~4 sigma) for efficiency
            dt_to_pulses = t_mid - pulse_centers
            mask = np.abs(dt_to_pulses) < 4 * sigma
            At = np.sum(A_0 * np.exp(-0.5 * (dt_to_pulses[mask] / sigma)**2))
            At_vec[i] = At
                
            H_mid = H_0 + (At * n_op)
            
            A = I + (((1j*dt) / (2*hbar)) * H_mid)
            B = I - (((1j*dt) / (2*hbar)) * H_mid)
            
            # Solve the system
            psi = np.linalg.solve(A, B @ psi)
            
            At_vec[i] = At
            # Probability of finding the system in the ground state (E_0), i.e. |0>
            P_0[i]    = np.abs(psi[0])**2
            # Probability of finding the system in the first excited state (E_1) i.e. |1>
            P_1[i]    = np.abs(psi[1])**2
            # Probability of finding the system in the second excited state (E_2) (leakage) (bad)
            P_2[i]    = np.abs(psi[2])**2

        self.t_vec, self.At_vec, self.P_0, self.P_1, self.P_2 = t_vec, At_vec, P_0, P_1, P_2
    
    def connectivity(self):
        s = ""
        s += "--- Circuit Connectivity -\n"
        # Access the vertices dictionary from the graph
        for label, node in self.circuit_graph.vertices.items():
            s += (f"Node '{label}':\n")
            for branch in node.branches:
                # Find the 'other' node in the branch
                other_node = [n for n in branch.nodes if n != node]
                connection = f"connected to {other_node[0].label}" if other_node else "grounded"
                if type(branch) is Capacitor:  
                    s += (f"  - {"Capacitor"} ({branch.C:.2e}) {connection}\n")
                elif type(branch) is Inductor:
                    s += (f"  - {"Inductor"} ({branch.L:.2e}) {connection}\n")
                else:
                    s += (f"  - {"Josephson Junction"} ({branch.EJ:.2e}) ({branch.C:.2e}) {connection}\n")
        return s
    
    def __repr__(self):
        return self.connectivity()
######################################## CIRCUIT CLASS ########################################

def plot_charge_distribution(n_cut: int, states: np.ndarray):
    # 1. Define the x-axis (number of Cooper pairs)
    n_vals = np.arange(-n_cut, n_cut + 1)
    
    # 2. Get the ground state (first column) and first excited state (second column)
    ground_state = states[:, 0]
    excited_state = states[:, 1]
    
    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plotting Ground State (|0>)
    plt.plot(n_vals, ground_state, 'o-', label='Ground State |0>', markersize=4)
    
    # Plotting First Excited State (|1>)
    plt.plot(n_vals, excited_state, 's-', label='First Excited State |1>', markersize=4)
    
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Number of Cooper Pairs (n)')
    plt.ylabel('Amplitude')
    plt.title('Transmon Wavefunctions in Charge Basis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_potential_energy(circuit: Circuit, min_flux: float, max_flux: float):
    node_phases = np.linspace(min_flux, max_flux, 400)
    
    # Potential energy [J]
    U = np.array([])
    
    # Calculate different values for potential energy
    # depending on node flux
    for node_phase in node_phases:
        # Convert phase (dimensionless) to flux (Webers)
        node_flux = node_phase * PHI_0
        U = np.append(U, circuit.get_potential_energy(np.array([node_flux])))
    
    # Plot potential energy
    plt.plot(node_phases, U / e / 1e-3, 'r', linewidth=3, label='Potential Energy')

    if circuit.energies.size > 0:
        # Overlay lowest energy eigenvalues
        for k in range(len(circuit.energies)):  # ground, first, second excited, etc.
            plt.hlines(circuit.energies[k] / e / 1e-3, xmin=min_flux, xmax=max_flux,
                    colors='k', linewidth=2, linestyles='-', label="Energy Level" if k == 0 else None)
    
    # Labels and formatting
    plt.xlabel(r'Phase $\phi$', fontsize=15)
    plt.ylabel('Energy (meV)', fontsize=15)
    plt.axis([min_flux, max_flux, None, None]) 
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.grid(alpha=0.3)
    plt.show()

def plot_pulse_sequence(t_vec: np.ndarray, At_vec: np.ndarray, num_steps=1000):
    plt.figure(figsize=(8, 4))
    plt.plot(t_vec[:num_steps] * 1e9, At_vec[:num_steps] / e / 1e-3, linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Pulse (meV)')
    plt.title('SFQ Pulse shape')
    plt.grid(True)
    plt.show()

def plot_rabi_oscillations(T_drive: float, t_vec: np.ndarray, P_0: np.ndarray, P_1: np.ndarray, P_2: np.ndarray):
    plt.figure(figsize=(8, 5))
    plt.plot(t_vec * 1e9, P_0, label='P0', linewidth=2)
    plt.plot(t_vec * 1e9, P_1, label='P1', linewidth=2)
    plt.plot(t_vec * 1e9, P_2, label='P2 (leakage)', linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Population')
    plt.title(f'SFQ-driven Rabi dynamics: f_drive = {(1/T_drive)/1e9:.3f} GHz')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
