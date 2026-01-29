from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, e
PHI_0 = hbar / (2 * e)

######################################## NODE CLASS ########################################
class Node:
    def __init__(self, label: str, branches: Optional[list[Branch]] = None):
        # e.g. "a", "b", etc.
        self.label = label
        # e.g [capacitor1, inductor1, JJ1, ...]
        self.branches = branches if branches != None else []

    def _get_degree(self):
        # number of connected capacitive branches
        capacitive_degree = 0
        # number of connected inductive branches
        inductive_degree = 0
        
        # NOTE: a JosephsonJunction is made up of a Capacitive branch and and Inductive branch
        for branch in self.branches:
            if type(branch) in [Capacitor, JosephsonJunction]:
                capacitive_degree += 1
            if type(branch) in [Inductor, JosephsonJunction]:
                inductive_degree += 1
                
        return capacitive_degree, inductive_degree    
######################################## NODE CLASS ########################################        

######################################## BRANCH CLASS ########################################
class Branch:
    def __init__(self, nodes: Optional[list[Node]] = None):
        self.nodes = nodes if nodes != None else []

class Capacitor(Branch):
    def __init__(self, value: float, nodes: Optional[list[Node]] = None):
        super().__init__(nodes)
        self.C = value
        
class Inductor(Branch):
    def __init__(self, value: float, nodes: Optional[list[Node]] = None):
        super().__init__(nodes)
        self.L = value
        
class JosephsonJunction(Branch):
    def __init__(self, EJ: float, CJ: float, nodes: Optional[list[Node]] = None):
        super().__init__(nodes)
        self.EJ = EJ
        self.C = CJ
######################################## BRANCH CLASS ########################################

######################################## GRAPH CLASS ########################################
class Graph:
    def __init__(self, vertices: list[Node], edges: list[Branch]):
        self.vertices = vertices
        self.edges = edges
######################################## GRAPH CLASS ########################################

######################################## CIRCUIT CLASS ########################################
class Circuit:
    """
    Circuit class to build and store relevant data for a circuit
    using the method of nodes.
    """
    def __init__(self, graph_rep: dict):
        self.labels                                           = graph_rep["nodes"]                     # ["a", "b", "c", etc.]
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
        self.n_hat, self.H_hat                                = self._quantize(n_cut=20)
        self.H_eigenvalues, self.H_eigenvectors               = self._diagonalize()
        self.D                                                = np.diag(self.H_eigenvalues)
        
                         
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
    
    def get_potential_energy(self, node_flux):
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
    def _quantize(self, n_cut: float):
        C_sum = self.capacitance_matrix[0][0]
        EJ    = self.josephson_junctions[0][2]
        
        # Cost to add one Cooper pair of charge
        EC = e**2 / (2*C_sum)

        # Define Hilbert space
        # e.g. [-20, -19, ..., 0, ..., 19, 20] for n_cut = 20
        # each n represents n Cooper pairs on the island
        n_vals = [i for i in range(-n_cut, n_cut+1)] 
        dim = len(n_vals)
        n_hat = np.diag(4 * EC * n_vals**2)
        
        # Build Hamiltonian Operator
        H = n_hat - 0.5 * EJ * (np.diag(np.ones(dim-1), 1) + np.diag(np.ones(dim-1), -1))
        
        return n_hat, H
    
    def _diagonalize(self):
        return np.linalg.eigh(self.H_hat)
        
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

def visualize_potential(circuit, flux_range=(-10e-15, 10e-15), num_points=100):
    """Plot the potential energy landscape"""
    phi_vals = np.linspace(*flux_range, num_points)
    
    # for every node in the circuit
    for i, node_label in enumerate(circuit.labels):
        # set the flux at that node to one of phi_vals, keeping all other nodes at 0
        # and store the resulting potential energy
        potentials = []
        for phi in phi_vals:
            node_flux = np.zeros(circuit.N)
            node_flux[i] = phi
            potentials.append(circuit.get_potential_energy(node_flux))
        
        plt.plot(phi_vals, potentials, label=f'Node {node_label}')
    
    plt.xlabel('Node Flux (Wb)')
    plt.ylabel('Potential Energy (J)')
    plt.title('Circuit Potential Energy Landscape')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Cooper Pair Box / Transmon-like circuit
    # Just a Josephson junction with shunt capacitance (no linear inductor)
    
    # Typical experimental values:
    EJ = 20e-23      # ~20 GHz Josephson energy (h * 20 GHz ≈ 1.3e-23 J)
    CJ = 5e-15       # 5 fF junction capacitance
    CS = 100e-15 # 100 fF shunt capacitance (transmon regime)
    
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
    
    for node in circuit.active_nodes:
        print(f"  Active node: {node.label}")
    for node in circuit.passive_nodes:
        print(f"  Passive node: {node.label}")
    
    print(circuit)
    print(f"Capacitance Matrix: {circuit.capacitance_matrix}")
    print(f"Inverse Inductance Matrix: {circuit.inv_inductance_matrix}")    
    
    PHI_0 = hbar / (2 * e)
    phi_test = np.zeros(circuit.N, dtype=float).reshape(-1)  # Force 1D
    phi_test[0] = 0.1 * PHI_0 
    q_test = (np.ones(circuit.N) * 1e-19).reshape(-1)  # Force 1D
    
    print(f"Potential Energy: {circuit.get_potential_energy(node_flux=phi_test)}")
    print(f"Kinetic Energy: {circuit.get_kinetic_energy(node_charge=q_test)}")
    print(f"Lagrangian: {circuit.get_lagrangian(node_flux=phi_test, node_charge=q_test)}")
    print(f"Normal Modes Squared: {circuit.normal_modes_squared}")
    print(f"Hamiltonian: {circuit.get_hamiltonian(node_flux=phi_test, node_charge=q_test)}")
    
    # visualize_potential(circuit=circuit)
    
    
if __name__=="__main__":
    main()