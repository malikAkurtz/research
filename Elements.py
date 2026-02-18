from __future__ import annotations
from typing import Optional
from scipy.constants import hbar, e
import numpy as np

# ---- Fundamental constant ----
PHI_0 = hbar / (2 * e)  # Reduced flux quantum [Wb]

######################################## NODE CLASS ########################################
class Node:
    def __init__(self, label: str, branches: Optional[list[Branch]] = None):
        # e.g. "a", "b", etc.
        self.label = label
        # e.g [capacitor1, inductor1, JJ1, ...]
        self.branches = branches if branches is not None else []

    def _get_degree(self):
        # number of connected capacitive branches
        capacitive_degree = 0
        # number of connected inductive branches
        inductive_degree = 0
        
        for branch in self.branches:
            if isinstance(branch, CapacitiveElement):
                capacitive_degree += 1
            if isinstance(branch, InductiveElement):
                inductive_degree += 1
                
        return capacitive_degree, inductive_degree    
######################################## NODE CLASS ######################################## 

######################################## BRANCH CLASS ########################################
# A BRANCH IS AN ELEMENT, TWO NODES CAN BE CONNECTED BY MULTIPLE BRANCHES
class Branch:
    def __init__(self, nodes: Optional[tuple[Node]] = None):
        self.nodes = nodes
        
    def calculate_energy(self):
        pass

class CapacitiveElement(Branch):
    def __init__(self, capacitance: float, nodes: Optional[tuple[Node]] = None):
        super().__init__(nodes)
        self.C = capacitance
        
    def calculate_energy(self, branch_charge: float, branch_charge_offset: float):
        return ((branch_charge - branch_charge_offset)**2) / (2 * self.C)
        
class Capacitor(CapacitiveElement):
    def __init__(self, capacitance: float, nodes: Optional[tuple[Node]] = None):
        super().__init__(capacitance, nodes)

class InductiveElement(Branch):
    def __init__(self, nodes: Optional[tuple[Node]] = None):
        super().__init__(nodes)
        
    def calculate_energy(self):
        pass
    
class Inductor(InductiveElement):
    def __init__(self, inductance: float, nodes: Optional[tuple[Node]] = None):
        super().__init__(nodes)
        self.L = inductance
        
    def calculate_energy(self, branch_flux: float, branch_flux_offset: float):
        return ((branch_flux - branch_flux_offset)**2) / (2 * self.L)
        
class JosephsonElement(InductiveElement):
    def __init__(self, josephson_energy: float, nodes: Optional[tuple[Node]] = None):
        super().__init__(nodes)
        self.EJ = josephson_energy
        
    def calculate_energy(self, branch_flux: float, branch_flux_offset: float):
        return -self.EJ * np.cos( (branch_flux - branch_flux_offset) / PHI_0)
        
######################################## BRANCH CLASS ########################################

######################################## GRAPH CLASS ########################################
class Graph:
    def __init__(self, vertices: list[Node], edges: list[Branch]):
        self.vertices = vertices
        self.edges = edges
######################################## GRAPH CLASS ########################################