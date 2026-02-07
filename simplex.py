from __future__ import annotations
import numpy as np
from Constraint import Constraint
from Objective import Objective
from foods import foods, food_keys
from constraints import diet_constraints, obj

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
        self.artifical_var_rows = []
        self.obj_flipped        = False
        
        self.A                  = None
        self.cur_basic_vars     = [None] * self.num_constraints
        
        self.optimal_obj_val    = None
        self.optimal_var_vals   = None
        
    def _verify_objective(self):
        # If the objective is to minimize, we have to transform it
        if self.objective.obj == "min":
            self.objective.coefficients = -1 * self.objective.coefficients
            self.objective.obj          = "max"
            self.obj_flipped            = True

    def _ensure_positve_rhs(self):
        for c in self.constraints:
            # 1) Ensure rhs >= 0
            if c.rhs < 0:
                # Multiply the rhs and coefficients by -1 and flip the type
                c.coefficients      *= -1
                c.rhs               *= -1
                c._flip_type()
    
    def _standardize(self):
        # 1) Initialize Tableu
        self.A = np.array([np.append(c.coefficients, c.rhs) for c in self.constraints])
        # 2) Iterate through constraints and add slack and artifical variables as appropriate
        for i in range(self.num_constraints):
            c = self.constraints[i]
            # *) Add artificial variables if type is equality
            if c.type == "=":
                # Add artifical variable and add it as a basic variable
                self._add_artifical_variable(idx=i)
            # *) Subtract slack variable add artifical variable is type is greater than
            elif c.type == ">=":
                # Subtract Slack variable
                self._subtract_slack_variable(idx=i)
                # Add artifical variable and add it as a basic variable
                self._add_artifical_variable(idx=i)
                # Change type to equality
                c.type = "="
            # *) Add slack variable if type is less than
            elif c.type == "<=":
                # Add slack variable and add it as a basic variable
                self._add_slack_variable(idx=i)
                # Change type to equality
                c.type = "="                
    
    def _add_artifical_variable(self, idx: int):
        new_col                  = np.array(np.zeros(len(self.A)))
        new_col[idx]             = 1.0
        self.A                   = np.insert(self.A, -1, np.array([new_col]), axis=1)
        self.artifical_var_idx.append(self.num_vars)
        self.cur_basic_vars[idx] = self.num_vars
        self.artifical_var_rows.append(idx)
        self.num_artifical_vars += 1
        self.num_vars           += 1
    
    def _add_slack_variable(self, idx: int):
        new_col              = np.array(np.zeros(len(self.A)))
        new_col[idx]         = 1.0
        self.A               = np.insert(self.A, -1, np.array([new_col]), axis=1)
        self.slack_var_idx.append(self.num_vars)
        self.cur_basic_vars[idx] = self.num_vars
        self.num_slack_vars += 1
        self.num_vars       += 1
    
    def _subtract_slack_variable(self, idx: int):
        new_col              = np.array(np.zeros(len(self.A)))
        new_col[idx]         = -1.0
        self.A               = np.insert(self.A, -1, np.array([new_col]), axis=1)
        self.slack_var_idx.append(self.num_vars)
        self.num_slack_vars += 1
        self.num_vars       += 1
        
    def _phase_1(self):
        # If we have artifical variables, we need to minimize their sum
        # i.e. maximize the sum of all negative artifical variables
        if self.num_artifical_vars > 0:
            # 1) Create the row for the phase 1 objective
            phase_1_obj_row = np.zeros(len(self.A[0]))
            for idx in self.artifical_var_idx:
                phase_1_obj_row[idx] = 1 # since maximizing negation
                
            # 2) Add the new row to thex bottom of our system
            self.A = np.append(self.A, np.array([phase_1_obj_row]), axis=0)
                   
            # 3) Clear their corresponding columns
            for row_idx, col_idx in zip(self.artifical_var_rows, self.artifical_var_idx):
                self._clear_col(pivot_row_idx=row_idx, pivot_col_idx=col_idx)
                
            # 4) Run simplex to minimize phase 1 objective
            optimal_vals, optimal_z = self._simplex(obj_row_idx=-1)
            if not np.isclose(optimal_z, 0):
                print("Problem is Unfeasible!")
                return
    
    def _phase_2(self):
        # 1) Add Phase 2 objective row to the bottom of A
        phase_2_obj_coefs     = np.append(self.objective.coefficients, np.zeros(self.num_slack_vars + self.num_artifical_vars))
        phase_2_obj_row       = np.append(phase_2_obj_coefs, 0.0)
        self.A = np.append(self.A, np.array([phase_2_obj_row]), axis=0)
        # Negate the Phase 2 objective row since we are maximizing
        self.A[-1] = -1 * self.A[-1]
        
        # 2) Get rid of artifical variable columns and Phase 1 objective row
        if self.num_artifical_vars > 0:
            # Get rid of artifical columns
            for idx in sorted(self.artifical_var_idx, reverse=True):
                self.A = np.delete(self.A, idx, axis=1)
                # Update cur_basic_vars indices that are affected by this deletion
                for i in range(len(self.cur_basic_vars)):
                    if self.cur_basic_vars[i] > idx:
                        self.cur_basic_vars[i] -= 1
                # Update slack variable indices that are affected by this deletion
                for i in range(len(self.slack_var_idx)):
                    if self.slack_var_idx[i] > idx:
                        self.slack_var_idx[i] -= 1
                self.num_vars           -= 1
                self.num_artifical_vars -= 1
                
            # Get rid of phase 1 objective row
            self.A = np.delete(self.A, -2, axis=0)
        
            # 3) Clear basic variables
            for i in range(self.num_constraints):
                basic_var = self.cur_basic_vars[i]
                self._clear_col(pivot_row_idx=i, pivot_col_idx=basic_var)
                
        # 4) Run simplex with actual objective
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
        
    def _simplex(self, obj_row_idx: int):        
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
                
                rhs_col = self.A[:, -1]

                # ratio test
                possible_vals = []
                for i in range(self.num_constraints):
                    if pivot_col[i] > 0:
                        possible_vals.append(rhs_col[i] / pivot_col[i])
                    else:
                        possible_vals.append(np.inf)
                        
                if all(val == np.inf for val in possible_vals):
                    print("Problem is unbounded!")
                    return None, None
                                        
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
                
        # During Phase 1, self.num_vars = num_decision_vars + num_slack_vars + num_artifical_vars
        # During Phase 2, self.num_vars = num_decision_vars + num_slack_vars
        optimal_vals = np.zeros(self.num_vars, dtype=float)
        
        for row_idx, var_idx in enumerate(self.cur_basic_vars):
            # Extract RHS for that basic variable row
            optimal_vals[var_idx] = self.A[:, -1][row_idx]
                
        optimal_z = self.A[obj_row_idx][-1]
        
        if self.obj_flipped:
            optimal_z *= -1
                        
        return optimal_vals, optimal_z

def main():
    
    system = System(
        num_decision_vars=len(foods),
        constraints=diet_constraints,
        objective=obj
        )
    
    # 1) Make sure we are maximizing an objective, transform if necessary
    system._verify_objective()
    print(system.A)
    # 2) Ensure the RHS >= 0
    system._ensure_positve_rhs()
    print(system.A)
    # 3) Put system in standard form
    system._standardize()
    print(system.A)
    # 4) Run Phase 1
    system._phase_1()
    print(system.A)
    # 5) Run Phase 2
    optimal_vals, optimal_z = system._phase_2()
    print(system.A)
    
    for idx, val in enumerate(optimal_vals):
        if val > 0:
            if idx < system.num_decision_vars:
                food_item = foods[food_keys[idx]]
                print(f"{food_item.name} = {val} {food_item.units}")
        
    print(f"Optimal Calorie Count: {optimal_z} ")
    
if __name__=="__main__":
    main()