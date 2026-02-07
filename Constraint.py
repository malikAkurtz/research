import numpy as np

class Constraint():
    def __init__(self, coefficients: np.ndarray, type: str, rhs: float):
        # len(coefficients) = # Decision Variables
        self.coefficients = coefficients
        self.type = type
        self.rhs = rhs
        
    def _flip_type(self):
        if self.type == "<=":
            self.type = ">="
        elif self.type == ">=":
            self.type = "<="
        # if self.type == "=":
            # do nothing