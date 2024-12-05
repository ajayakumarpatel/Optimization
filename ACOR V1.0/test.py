import ACOR_Optimizer
import numpy as np
import math

def camelback(x, y):
    return (4 - 2.1 * x**2 + (x**4) / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2

def ackley(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.e

def goldstein_price(x, y):
    term1 = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
    term2 = (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return term1 * term2

def Himmelblau_func(x,y):
    return (((x**2+y-11)**2) + (((x+y**2-7)**2))) ##4 global minimum

def easom_func(x1, x2):
    return -(np.cos(x1)*np.cos(x2))*np.exp(-(x1-math.pi)**2-(x2-math.pi)**2)

def sphere_func(x1, x2, x3):
    return x1**2 + x2**2 + x3**2

def rastrigin_func(X1, X2, X3):
    return 10 * 3 + (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2)) + (X3**2 - 10 * np.cos(2 * np.pi * X3))


if __name__ == "__main__": 
    search_space = [(-20, 20), (-20, 20)]
    aco_r = ACOR_Optimizer.ACOR(obj_function = goldstein_price, search_space = search_space)
    result = aco_r.optimizer()
    print(f' Minimum Value of the function: {result[0]} \n Best point: {result[1]}')