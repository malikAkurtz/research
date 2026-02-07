import numpy as np

class Objective():
    def __init__(self, coefficients: np.ndarray, obj: str):
        self.coefficients = coefficients
        self.obj          = obj