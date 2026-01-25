from __future__ import annotations
from typing import Optional
import numpy as np

######################################## NODE CLASS ########################################
class Node:
    def __init__(self, label: str, branches: Optional[list[Branch]] = None, active: Optional[bool] = None):
        self.label = label
        self.branches = branches if branches != None else []
        self.active = active
        
    def _get_degree(self):
        capacitive_degree = 0
        inductive_degree = 0
        
        for branch in self.branches:
            if branch.type == "C":
                capacitive_degree += 1
            else:
                inductive_degree += 1
                
        return capacitive_degree, inductive_degree    
######################################## NODE CLASS ########################################        

######################################## BRANCH CLASS ########################################
class Branch:
    def __init__(self, type: str, value: float, nodes: Optional[list[Node]] = None):
        self.type = type
        self.value = value
        self.nodes = nodes if nodes != None else []
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
        self.labels                                           = ["gnd"] + graph_rep["nodes"]
        self.circuit_graph                                    = Circuit._build_master_graph(graph_rep)
        self.capacitive_sub_graph, self.inductive_sub_graph   = self._build_sub_graphs()
        self.active_nodes, self.passive_nodes                 = self._partition_nodes()
        self.N                                                = len(self.active_nodes)
        self.P                                                = self.N + len(self.passive_nodes)
        self.capacitance_matrix, self.inv_inductance_matrix   = self._build_matrices()
        self.offset_matrix                                    = self._build_offset_matrix(graph_rep)
    
    def _build_offset_matrix(self, graph_rep):
        matrix = np.zeros((self.P, self.P))
        
        for offset in graph_rep["external_flux"]:
            node1_label = offset[0]
            node2_label = offset[1]
            value = offset[2]
            
            j = self.labels.index(node1_label)
            k = self.labels.index(node2_label)
            
            matrix[j][k] = value
            matrix[k][j] = value
            
        matrix = np.delete(matrix, 0, axis=0)
        matrix = np.delete(matrix, 0, axis=1)
            
        return matrix
                         
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
            capacitor = graph_rep['capacitors'][i]
            node1 = nodes[capacitor[0]]
            node2 = nodes[capacitor[1]]
            value = capacitor[2]
            
            # create the branch object with nodes as arguments
            branch = Branch(type="C", value=value, nodes=[node1, node2])
            
            branches.append(branch)
            node1.branches.append(branch)
            node2.branches.append(branch)
            
        # create and assign inductive branches
        for i in range(len(graph_rep['inductors'])):
            # save the inductor tuple
            inductor = graph_rep['inductors'][i]
            node1 = nodes[inductor[0]]
            node2 = nodes[inductor[1]]
            value = inductor[2]
            
            # create the branch object with nodes as arguments
            branch = Branch(type="L", value=value, nodes=[node1, node2])
            
            branches.append(branch)
            node1.branches.append(branch)
            node2.branches.append(branch)
            
        # break symmetry
        for node in nodes.values():
            if node_label == "gnd":
                continue
            capacitive_element = False
            for branch in node.branches:
                if branch.type == "C":
                    capacitive_element = True
                    break
            if not capacitive_element:
                capacitive_branch = Branch(type="C", value=1e-20, nodes=[node, nodes["gnd"]])
                
                branches.append(capacitive_branch)
                node.branches.append(capacitive_branch)
                nodes["gnd"].branches.append(capacitive_branch)
                        
        return Graph(nodes, branches)
            
    def _build_sub_graphs(self) -> dict:
        nodes    = self.circuit_graph.vertices
        branches = self.circuit_graph.edges
        
        capacitive_branches = []
        inductive_branches  = []
        
        # filter  branches
        for branch in branches:
            if branch.type == "C":
                capacitive_branches.append(branch)
            else:
                inductive_branches.append(branch)
        
        return Graph(nodes, capacitive_branches), Graph(nodes, inductive_branches)
    
    def _partition_nodes(self):
        active_nodes    = []
        passive_nodes   = []
        
        print(self.circuit_graph.vertices)
        # assuming symmetry already broken, only have to check inductive degree
        for node_label, node in self.circuit_graph.vertices.items():
            capacitive_degree, inductive_degree = node._get_degree()
            if (inductive_degree >= 0):
                active_nodes.append(node)
            else:
                passive_nodes.append(node)
                
        return active_nodes, passive_nodes
    
    def _build_matrices(self):
        capacitance_matrix      = np.zeros((self.P, self.P))
        inv_inductance_matrix   = np.zeros((self.P, self.P))
        
        for branch in self.circuit_graph.edges:
            node1_label = branch.nodes[0].label
            node2_label = branch.nodes[1].label
            
            j = self.labels.index(node1_label)
            k = self.labels.index(node2_label)
            
            if branch.type == "C":
                capacitance_matrix[j][k] += -branch.value
                capacitance_matrix[k][j] += -branch.value
            else:
                inv_inductance_matrix[j][k] += -1 / branch.value
                inv_inductance_matrix[k][j] += -1 / branch.value
                
        for j in range(self.P):
            capacitance_matrix[j][j]    = -1 * np.sum(capacitance_matrix[j]).item()
            inv_inductance_matrix[j][j] = -1 * np.sum(inv_inductance_matrix[j]).item()
            
        capacitance_matrix = np.delete(capacitance_matrix, 0, axis=0)
        capacitance_matrix = np.delete(capacitance_matrix, 0, axis=1)
        
        inv_inductance_matrix = np.delete(inv_inductance_matrix, 0, axis=0)
        inv_inductance_matrix = np.delete(inv_inductance_matrix, 0, axis=1)
    
        return capacitance_matrix, inv_inductance_matrix
    
    def get_kinetic_energy(self, node_flux_dot):
        return (node_flux_dot.T @ self.capacitance_matrix @ node_flux_dot) / 2
    
    def get_potential_energy(self, node_flux, T):
        term_1 = (node_flux.T @ self.inv_inductance_matrix @ node_flux) / 2
        
        term_2 = 0
        for branch in self.circuit_graph:
            if branch not in T.edges:
                node1_label = branch.nodes[0].label
                node2_label = branch.nodes[1].label
                
                j = self.labels.index(node1_label)
                k = self.labels.index(node2_label)
                
                term_2 += ((node_flux[j] - node_flux[k]) * self.offset_matrix[j][k]) / branch.value
        
        return term_1 + term_2
    
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
                s += (f"  - {branch.type} ({branch.value:.2e}) {connection}\n")
        return s
    
    def __repr__(self):
        return self.connectivity()
######################################## CIRCUIT CLASS ########################################

    # Black box function, straight from Gemini
    def _build_min_spanning_tree(self):
        root = self.circuit_graph.vertices["gnd"]
        visited_nodes = {root}
        tree_edges = []
        
        # We use a queue for BFS: (current_node)
        queue = [root]
        
        # Get all edges, but prioritize Capacitors over Inductors
        # This makes the 'Tree' as capacitive as possible
        all_edges = sorted(self.circuit_graph.edges, 
                           key=lambda b: 0 if b.type == "C" else 1)

        while queue:
            current_node = queue.pop(0)
            
            # Look at all branches connected to this node
            for branch in current_node.branches:
                # Find the 'other' node in the branch
                other_node = branch.nodes[1] if branch.nodes[0] == current_node else branch.nodes[0]
                
                if other_node not in visited_nodes:
                    # This branch is now part of our spanning tree
                    visited_nodes.add(other_node)
                    tree_edges.append(branch)
                    queue.append(other_node)
        
        # The vertices of the tree are the same as the original graph
        return Graph(list(self.circuit_graph.vertices.values()), tree_edges)
    
def main():
    graph_rep = \
    {
        'nodes': ['a', 'b', 'c'],
        'capacitors': [
            ('a', 'gnd', 1e-12),
            ('b', 'gnd', 2e-12),
            ('a', 'b', 0.5e-12),
        ],
        'inductors': [
            ('a', 'b', 1e-9),
            ('b', 'c', 2e-9),
        ],
        'external_flux': {
            ('a', 'b'): 1e-15
        }
    }
    
    circuit = Circuit(graph_rep)
    
    print(circuit)
    print(circuit.inv_inductance_matrix)
    print(circuit.capacitance_matrix)
    
if __name__=="__main__":
    main()