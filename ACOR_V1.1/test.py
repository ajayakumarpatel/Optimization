import ACOR_Optimizer_MultiThread
import dummy
import hybrid_cpu_gpu
import run_simulation
import numpy as np
import math
import time

# Main script
if __name__ == "__main__":
    start_time = time.time()
    # search_space = [(-20, 20), (-20, 20)]
    search_space_json = "param.json"
    simulation = run_simulation.run_simulation()  # Get the Simulation instance

    aco_r = ACOR_Optimizer_MultiThread.ACOR(obj_function=simulation.camelback, search_space_json=search_space_json, n_ants =20, new_ants = 2, n_cycles = 120, n_local_cycles = 5)
    # aco_r = hybrid_cpu_gpu.ACOR(obj_function=simulation.camelback, search_space=search_space, n_ants = 20, new_ants = 2, n_cycles = 120, n_local_cycles = 5)
    # aco_r = dummy.ACOR(obj_function=simulation.camelback, search_space=search_space_json, n_ants = 20, new_ants = 2, n_cycles = 120, n_local_cycles = 5)
    result = aco_r.optimizer()
    end_time = time.time()
    print(f'Minimum Value of the function: {result[0]} \n Best point: {result[1]}')
    print((end_time - start_time)*1e3)

