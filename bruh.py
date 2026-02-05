import numpy as np

def main():
    tableu = np.array([
        [2, 0, -1, 1, 0, 2, 0, -2],
        [-1, 1, -1, 0, 0, 1, 0, 1],
        [2, 0, 1, -1, 0, -1, 1, 2],
        [3, 0, 1, 0, 1, -1, 0, 1]
    ])
    
    pivot_row = tableu[3]
    
    tableu[0] = tableu[0] + pivot_row
    tableu[1] = tableu[1] + (pivot_row)
    tableu[2] = tableu[2] + (-1*pivot_row)
    
    print(tableu)
    
    
if __name__=="__main__":
    main()