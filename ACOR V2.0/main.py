import numpy as np
from ACOR_Optimizer import ACOR
from numba import cuda
import time

if __name__=="__main__":
    if(cuda.is_available()==True):
        search_space = [(-20, 20), (-20, 20)]
        s_time = time.time()
        aco_r = ACOR(search_space = search_space)
        result = aco_r.run()
        e_time = time.time()
        print(f' Minimum Value of the function: {result[0]} \n Best point: {result[1]}')
        print(f'execution time: {(e_time - s_time)*1e3} ms')
    else:
        print('Error: GPU Not available!!')