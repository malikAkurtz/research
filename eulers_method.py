import numpy as np


def main():
    x_0 = 0
    v_0 = 1
    
    dt = 0.001
    
    T = 0
    x_prev = x_0
    v_prev = v_0
    
    while T < np.pi:
        x_new = x_prev + v_prev * dt
        v_new = v_prev - x_prev * dt
        
        x_prev = x_new
        v_prev = v_new
        
        T += dt
    
    print(T)
    print(x_new)
    
if __name__=="__main__":
    main()