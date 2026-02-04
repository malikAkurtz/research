import numpy as np


def simplex(tableu: np.ndarray, num_basic_vars: int):
    # assuming basic var cols are after non-basic
    # assuming z row is tableu[0]
    
    # total number of columns i.e. total number variables
    N = len(tableu[0]) - 1 # - 1 for rhs column
    # total number of decision variables
    n = N - num_basic_vars
    
    # list of indices corresponding to the basic variables at any given moment
    basic_vars = [i for i in range(n, N)]

    new_tableu = tableu.copy()
    
    while True:
        # Find the new basic variable
        most_negative_col_idx = 0
        most_negative_val = new_tableu[0][most_negative_col_idx]
        
        for i in range(N):
            z_row = new_tableu[0]
            rhs_col = new_tableu[:, -1]
            candidate_val = z_row[i]
            
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
            pivot_col = new_tableu[:, pivot_col_idx]

            # ratio test
            possible_vals = [rhs_col[i] / pivot_col[i] if not (rhs_col[i] / pivot_col[i] > 0) else np.inf for i in range(1, len(pivot_col))]
            
            leaving_basic_var_idx = np.argmin(possible_vals)            
            basic_vars[leaving_basic_var_idx] = pivot_col_idx
            
            # clear the column corresponding to the new basic variable
            pivot_row_idx = leaving_basic_var_idx + 1 # to account for the z row in the tableu
            pivot_value = new_tableu[pivot_row_idx][pivot_col_idx]
            
            new_tableu[pivot_row_idx] = new_tableu[pivot_row_idx]  / pivot_value # pivot_value = 1
            
            pivot_row = new_tableu[pivot_row_idx]
            
            for i in range(num_basic_vars + 1):
                if i == pivot_row_idx:
                    continue
                else:
                    row_value = new_tableu[i][pivot_col_idx]
                    new_tableu[i] = new_tableu[i] + (-row_value * pivot_row)
            
    # idx 0 <=> x_0, idx 1 <=> x_1, etc
    optimal_vals = np.zeros(n, dtype=float)
    
    for idx, var_idx in enumerate(basic_vars):
        # x_0, ..., x_n-1
        if var_idx < n:
            optimal_vals[var_idx] = new_tableu[:, -1][idx + 1]
            
    optimal_z = new_tableu[0][-1]
        
    return optimal_vals, optimal_z

def main():
    inital_tableu = np.array([
        [-3.0, -2.0, -4.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 4.0],
        [2.0, 0.0, 3.0, 0.0, 1.0, 0.0, 5.0],
        [2.0, 1.0, 3.0, 0.0, 0.0, 1.0, 7.0]
    ])
    
    optimal_vals, optimal_z = simplex(tableu=inital_tableu, num_basic_vars=3)
    
    for idx, val in enumerate(optimal_vals):
        print(f"x_{idx} = {val}")
        
    print(f"Optimal z Value: {optimal_z} ")
    
if __name__=="__main__":
    main()