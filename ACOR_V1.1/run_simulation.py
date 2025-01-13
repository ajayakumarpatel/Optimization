import numpy as np
import math

class Simulation:
    def __init__(self):
        self.count = 0
    def easom_func(self, x1, x2):
        return -(np.cos(x1)*np.cos(x2))*np.exp(-(x1-math.pi)**2-(x2-math.pi)**2)
    
    def rastrigin_func(self, X1, X2, X3):
        return 10 * 3 + (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2)) + (X3**2 - 10 * np.cos(2 * np.pi * X3))
    
    def sphere_func(self, x1, x2, x3):
        return x1**2 + x2**2 + x3**2
    
    def camelback(self, x, y):
        return (4 - 2.1 * x**2 + (x**4) / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2
    
    def ackley(self, x, y):
        term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        return term1 + term2 + 20 + np.e


def run_simulation():
    return Simulation()
