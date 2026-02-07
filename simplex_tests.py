import numpy as np
from Simplex import System
from Objective import Objective
from Constraint import Constraint

def test_1():
    obj = Objective(coefficients=[3.0, 2.0, 4.0], obj="max")
    c0 = Constraint(
        coefficients=np.array([1.0, 1.0, 2.0]),
        type="<=",
        rhs=4.0
        )
    c1 = Constraint(
        coefficients=np.array([2.0, 0.0, 3.0]),
        type="<=",
        rhs=5.0
        )
    c2 = Constraint(
        coefficients=np.array([2.0, 1.0, 3.0]),
        type="<=",
        rhs=7.0
        )
    system = System(
        num_decision_vars=3,
        constraints=[c0, c1, c2],
        objective=obj
        )
    # 1) Make sure we are maximizing an objective, transform if necessary
    system._verify_objective()
    # 2) Ensure the RHS >= 0
    system._ensure_positve_rhs()
    # 3) Put system in standard form
    system._standardize()
    # 4) Run Phase 1
    system._phase_1()
    # 5) Run Phase 2
    optimal_vals, optimal_z = system._phase_2()
    
    assert (np.allclose(optimal_vals, [2.5, 1.5, 0.0, 0.0, 0.0, 0.5]) and optimal_z == 10.5), "Test 1 Failed"
    
def test_2():
    obj = Objective(coefficients=np.array([6.0, 3.0]), obj="min")
    c0 = Constraint(
        coefficients=np.array([1.0, 1.0]),
        type=">=",
        rhs=1.0
        )
    c1 = Constraint(
        coefficients=np.array([2.0, -1.0]),
        type=">=",
        rhs=1.0
        )
    c2 = Constraint(
        coefficients=np.array([0.0, 3.0]),
        type="<=",
        rhs=2.0
        )
    system = System(
        num_decision_vars=2,
        constraints=[c0, c1, c2],
        objective=obj
        )
    # 1) Make sure we are maximizing an objective, transform if necessary
    system._verify_objective()
    # 2) Ensure the RHS >= 0
    system._ensure_positve_rhs()
    # 3) Put system in standard form
    system._standardize()
    # 4) Run Phase 1
    system._phase_1()
    # 5) Run Phase 2
    optimal_vals, optimal_z = system._phase_2()
    
    assert (np.allclose(optimal_vals, [(2/3), (1/3), 0.0, 0.0, 1.0]) and optimal_z == 5), "Test 2 Failed"


def main():
    test_1()
    print("Test 1 Passed")
    test_2()
    print("Test 2 Passed")

if __name__=="__main__":
    main()