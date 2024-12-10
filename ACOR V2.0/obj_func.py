import numpy as np
from numba import cuda
import cupy as cp
import math
import time


@cuda.jit(device=True)
def run_simulation(x, y):
    print('run...')
    count = 0
    for i in range(10000):
        for j in range(10000):
            if(i==j):
                count = 0
            else:
                count += math.sqrt(i**2 +j**2)

    return (4 - 2.1 * x**2 + (x**4) / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2

# s_time = time.time()
# r = run_simulation(2,4)
# e_time = time.time()
# print(f'exe time: {e_time - s_time} sec')
# print(cp.__version__)
