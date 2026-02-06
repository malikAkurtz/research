from __future__ import annotations
import numpy as np
from Food import Food
from food_dict import foods

class System():
    def __init__(self, num_decision_vars: int, constraints: list[Constraint], objective: Objective):
        self.num_decision_vars  = num_decision_vars
        self.num_vars           = num_decision_vars
        self.num_slack_vars     = 0
        self.num_artifical_vars = 0
        self.num_constraints    = len(constraints)
        self.constraints        = constraints
        self.objective          = objective
        self.slack_var_idx      = []
        self.artifical_var_idx  = []
        self.artifical_vars     = False
        self.obj_flipped        = False
        self.A                  = None
        self.cur_basic_vars     = []
        
    def _verify_objective(self):
        # If the objective is to minimize, we have to transform it
        if self.objective.obj == "min":
            self.objective.coefficients = -1 * self.objective.coefficients
            self.objective.obj           = "max"
            self.obj_flipped             = True

    def _ensure_positve_rhs(self):
        for c in self.constraints:
            # 1) Ensure rhs >= 0
            if c.rhs < 0:
                # Multiply the rhs and coefficients by -1 and flip the type
                c.coefficients      *= -1
                c.rhs               *= -1
                c._flip_type()
    
    def _standardize(self):
        self.A = np.array([np.append(c.coefficients, c.rhs) for c in self.constraints])
        # Add phase 2 objective row
        objective_row = np.array([np.array(np.append(self.objective.coefficients, 0.0))])
        self.A = np.append(self.A, objective_row, axis=0)
        # Negate the phase 2 objective row since we are maximizing
        self.A[-1] = -1 * self.A[-1]
        # For each constraint
        for i in range(self.num_constraints):
            c            = self.constraints[i]
            row          = self.A[i]
            # 1) Add artificial variables if type is equality
            if c.type == "=":
                # Add artifical variable
                self._add_artifical_variable(idx=i)
            # 2) Subtract slack variable, add artifical variable
            elif c.type == ">=":
                # Subtract Slack variable
                self._subtract_slack_variable(idx=i)
                # Add artifical variable
                self._add_artifical_variable(idx=i)
                # Change type to equality
                c.type = "="
            # 3) Add slack variable
            elif c.type == "<=":
                # Add slack variable
                self._add_slack_variable(idx=i)
                # Change type to equality
                c.type = "="                
                
    def _phase_1(self):
        # Create phase 1 pseudo objective
        # Want to minimize the sum of all artifical variables
        # i.e. maximize the sum of all negative artifical variables
        if self.artifical_var_idx:
            # 1) Create the row for the phase 1 objective
            phase_1_obj_row = np.zeros(self.num_vars + 1) # + 1 for the rhs column
            for idx in self.artifical_var_idx:
                phase_1_obj_row[idx] = 1 # since minimizing
                
            # 2) Add the new row to our system
            self.A = np.append(self.A, np.array([phase_1_obj_row]), axis=0)
                   
            # 3) Clear their corresponding columns
            for i, idx in enumerate(self.artifical_var_idx):
                self._clear_col(pivot_row_idx=i, pivot_col_idx=idx)
                
            # 4) Run simplex to minimize psuedo objective
            optimal_vals, optimal_z = self._simplex(obj_row_idx=-1)
            if optimal_z != 0:
                print("Problem is Unfeasible!")
                return
    
    def _phase_2(self):
        # 1) Get rid of artifical variable columns and phase 1 objective row
        if self.num_artifical_vars > 0:
            # Get rid of artifical columns
            for idx in sorted(self.artifical_var_idx, reverse=True):
                self.A = np.delete(self.A, idx, axis=1)
                # Update cur_basic_vars indices that are affected by this deletion
                for i in range(len(self.cur_basic_vars)):
                    if self.cur_basic_vars[i] > idx:
                        self.cur_basic_vars[i] -= 1
                self.num_vars           -= 1
                self.num_artifical_vars -= 1
                
            # Get rid of phase 1 objective row
            self.A = np.delete(self.A, -1, axis=0)
        
            # 2) Clear basic variables
            print(self.cur_basic_vars)
            for i in range(self.num_constraints):
                basic_var = self.cur_basic_vars[i]
                if basic_var < self.num_vars:
                    self._clear_col(pivot_row_idx=i, pivot_col_idx=basic_var)
        # Run simplex with actual objective
        return self._simplex(obj_row_idx=-1)
        
    def _clear_col(self, pivot_row_idx: int, pivot_col_idx: int):
        pivot_row = self.A[pivot_row_idx]
        
        for i in range(len(self.A)): # now including phase 1 and 2 objectives rows at the bottom
            if i == pivot_row_idx:
                continue
            else:
                row_value = self.A[i][pivot_col_idx]
                if row_value != 0:
                    self.A[i] = self.A[i] + (-row_value*pivot_row)
        
    def _add_artifical_variable(self, idx: int):
        new_col                  = np.array(np.zeros(len(self.A)))
        new_col[idx]             = 1.0
        self.A                   = np.insert(self.A, -1, np.array([new_col]), axis=1)
        self.artifical_var_idx.append(self.num_vars)
        self.num_artifical_vars += 1
        self.num_vars           += 1
    
    def _add_slack_variable(self, idx: int):
        new_col              = np.array(np.zeros(len(self.A)))
        new_col[idx]         = 1.0
        self.A               = np.insert(self.A, -1, np.array([new_col]), axis=1)
        self.slack_var_idx.append(self.num_vars)
        self.num_slack_vars += 1
        self.num_vars       += 1
    
    def _subtract_slack_variable(self, idx: int):
        new_col              = np.array(np.zeros(len(self.A)))
        new_col[idx]         = -1.0
        self.A               = np.insert(self.A, -1, np.array([new_col]), axis=1)
        self.slack_var_idx.append(self.num_vars)
        self.num_slack_vars += 1
        self.num_vars       += 1
        
    def _simplex(self, obj_row_idx: int):
        # assuming basic var cols are after non-basic
        n = self.num_vars
        m = self.num_constraints
        num_basic_vars = n - m
        print(f"n: {n}")
        print(f"m: {m}")
        print(f"num_basic_vars: {num_basic_vars}")        
        
        while True:
            # Find the new basic variable
            most_negative_col_idx = 0
            most_negative_val = self.A[obj_row_idx][most_negative_col_idx]
            
            for i in range(self.num_vars):
                obj_row = self.A[obj_row_idx]
                rhs_col = self.A[:, -1]
                candidate_val = obj_row[i]
                
                if candidate_val < most_negative_val:
                    most_negative_val = candidate_val
                    most_negative_col_idx = i
                    
            # if after checking all columns, there are no negative values
            if most_negative_val >= 0:
                # then we are done running the Simplex algorithm
                break
            # Otherwise, we need to alter the basic variables, clear the 
            else: 
                pivot_col_idx = most_negative_col_idx
                pivot_col = self.A[:, pivot_col_idx]
                print(f"pivot_col_idx: {pivot_col_idx}")
                
                rhs_col = self.A[:, -1]

                # ratio test
                possible_vals = []
                for i in range(m):
                    if pivot_col[i] <= 0:
                        possible_vals.append(np.inf)
                    else:
                        possible_vals.append(rhs_col[i] / pivot_col[i])
                        
                print(f"possible_vals: {possible_vals}")
                
                leaving_basic_var_idx = np.argmin(possible_vals)            
                self.cur_basic_vars[leaving_basic_var_idx] = pivot_col_idx
                
                # clear the column corresponding to the new basic variable
                pivot_row_idx = leaving_basic_var_idx
                pivot_value = self.A[pivot_row_idx][pivot_col_idx]
                
                self.A[pivot_row_idx] = self.A[pivot_row_idx]  / pivot_value # pivot_value = 1
                
                pivot_row = self.A[pivot_row_idx]
                
                for i in range(len(self.A)):
                    if i == pivot_row_idx:
                        continue
                    else:
                        row_value = self.A[i][pivot_col_idx]
                        self.A[i] = self.A[i] + (-row_value * pivot_row)
                
        # idx 0 <=> x_0, idx 1 <=> x_1, etc
        optimal_vals = np.zeros(n, dtype=float)
        
        for idx, var_idx in enumerate(self.cur_basic_vars):
            # x_0, ..., x_n-1
            if var_idx < n:
                optimal_vals[var_idx] = self.A[:, -1][idx]
                
        optimal_z = self.A[obj_row_idx][-1]
        
        if self.obj_flipped:
            optimal_z *= -1
            
        return optimal_vals, optimal_z

class Objective():
    def __init__(self, coefficients: np.ndarray, obj: str):
        self.coefficients = coefficients
        self.obj          = obj
    
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

def main():
    # obj = Objective(coefficients=[3.0, 2.0, 4.0], obj="max")
    # c0 = Constraint(
    #     coefficients=np.array([1.0, 1.0, 2.0]),
    #     type="<=",
    #     rhs=4.0
    #     )
    # c1 = Constraint(
    #     coefficients=np.array([2.0, 0.0, 3.0]),
    #     type="<=",
    #     rhs=5.0
    #     )
    # c2 = Constraint(
    #     coefficients=np.array([2.0, 1.0, 3.0]),
    #     type="<=",
    #     rhs=7.0
    #     )
    # system = System(
    #     num_decision_vars=3,
    #     constraints=[c0, c1, c2],
    #     objective=obj
    #     )
    # obj = Objective(coefficients=np.array([6.0, 3.0]), obj="min")
    # c0 = Constraint(
    #     coefficients=np.array([1.0, 1.0]),
    #     type=">=",
    #     rhs=1.0
    #     )
    # c1 = Constraint(
    #     coefficients=np.array([2.0, -1.0]),
    #     type=">=",
    #     rhs=1.0
    #     )
    # c2 = Constraint(
    #     coefficients=np.array([0.0, 3.0]),
    #     type="<=",
    #     rhs=2.0
    #     )
    # system = System(
    #     num_decision_vars=2,
    #     constraints=[c0, c1, c2],
    #     objective=obj
    #     )
    calorie_coefs = [f.calories for f in foods]
    obj = Objective(
        coefficients=np.array(calorie_coefs), 
        obj="min")
    
    carb_coefs = [f.carbs for f in foods]
    carb_constraint = Constraint(
        coefficients=np.array(carb_coefs),
        type=">=",
        rhs=150.0
        )
    
    fat_coefs = [f.fat for f in foods]
    fat_constraint = Constraint(
        coefficients=np.array(fat_coefs),
        type=">=",
        rhs=80.0
        )
    
    protein_coefs = [f.protein for f in foods]
    protein_constraint = Constraint(
        coefficients=np.array(protein_coefs),
        type=">=",
        rhs=170.0
        )
    system = System(
        num_decision_vars=len(foods),
        constraints=[
            carb_constraint,
            fat_constraint,
            protein_constraint,
            ],
        objective=obj
        )
    
    # 1) Make sure we are maximizing an objective, transform if necessary
    system._verify_objective()
    # 2) Ensure the RHS >= 0
    system._ensure_positve_rhs()
    # Put system in standard form
    system._standardize()
    print(f"System After Standardization: ")
    print(system.A)
    # 2) Run Phase 1
    system._phase_1()
    print(f"System After Phase 1: ")
    print(system.A)
    # 3) Run Phase 2
    optimal_vals, optimal_z = system._phase_2()
    print(f"System After Phase 2: ")
    print(system.A)
    
    for idx, val in enumerate(optimal_vals):
        print(f"x_{idx} = {val}")
        
    print(f"Optimal z Value: {optimal_z} ")
    
if __name__=="__main__":
    main()