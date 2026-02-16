from __future__ import annotations
from typing import Optional

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
        
        # NOTE: a JosephsonJunction is made up of a Capacitive branch and and Inductive branch
        for branch in self.branches:
            if isinstance(branch, (Capacitor, JosephsonJunction)):
                capacitive_degree += 1
            if isinstance(branch, (Inductor, JosephsonJunction)):
                inductive_degree += 1
                
        return capacitive_degree, inductive_degree    
######################################## NODE CLASS ######################################## 

######################################## BRANCH CLASS ########################################
class Branch:
    def __init__(self, nodes: Optional[list[Node]] = None):
        self.nodes = nodes if nodes is not None else []

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