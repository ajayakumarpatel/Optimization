"""
Code Title: Modified Ant Colony Optimization (ACOR) Algorithm
Version: 1.0
Developed By: Ajaya Kumar Patel
Release Date: 

Description:
This script implements a modified version of Ant Colony Optimization (ACOR) algorithm for solving optimization problems. 
The algorithm is inspired by the behavior of ants in finding optimal paths.

Features:
- Parallel evaluation of solutions using multi processing.
- Customizable search space and objective function.

Contact:
For feedback or inquiries, please contact: [ajayakumarpatel@agnikul.in]
"""

import multiprocessing as mp

import os
import csv
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
import Run_Simulation
import random
import time

output_path = '/home/newton/code/Agnibaan/Ajaya_OPTIM/output/'
time_stamp = time.time()
formatted_time = datetime.fromtimestamp(time_stamp).strftime('%H%M%S_%d%m%y')
print(f'Time-Date: {formatted_time}')

def save_archive_to_csv(archive, filename):
    with open(output_path+filename, mode = 'w', newline='') as file:
        writer = csv.writer(file)
        header = ["Fitness"] + [f"Variable_{i}" for i in range(1, len(archive[0][1]) + 1)]
        writer.writerow(header)

        for solution in archive:
            fitness = solution[0]
            variables = solution[1]
            row = [fitness] + variables
            writer.writerow(row)


class ACOR:
    def __init__(self, obj_function, search_space_json, n_ants = 20, new_ants = 2, n_cycles = 400, n_local_cycles = 5, q = 0.5, elite_count = 1, elite_weight = 0.4, evaporation_rate = 0.7, rocket_json = 'Input_JSON/vehicle.json'):
        
        print(f' Archive Size: {n_ants} \n number of new ants: {new_ants} \n Number of cycle: {n_cycles} \n Number of Local Cycle: {n_local_cycles}')
        print(f' elite count: {elite_count}, elite_weight: {elite_weight}')
        print(f' Evaporation Rate: {evaporation_rate}')
        with open(search_space_json, 'r') as file:
            search_space_data = json.load(file)
        
        with open(rocket_json, 'r') as file:
            rocket_param_data = json.load(file)
        
        self.obj_function = obj_function

        self.search_space = [
            search_space_data['Trajectory_Parameter_Inputs']['POM1_vec_int'],
            search_space_data['Trajectory_Parameter_Inputs']['POM1_duration_int'],
            search_space_data['Trajectory_Parameter_Inputs']['POM2_vec_int'],
            search_space_data['Trajectory_Parameter_Inputs']['GravTurn_eps'],
            search_space_data['Trajectory_Parameter_Inputs']['BLT1_param_int'],
            search_space_data['Trajectory_Parameter_Inputs']['GravTurn_end_alt_int'],
            search_space_data['Trajectory_Parameter_Inputs']['GT_Coasting_Time_int'],
            search_space_data['Trajectory_Parameter_Inputs']['Vertical_end_alt'],
            search_space_data['Trajectory_Parameter_Inputs']['BLT1_a'],
            search_space_data['Trajectory_Parameter_Inputs']['BLT_Coasting_Time_int'],
            search_space_data['Trajectory_Parameter_Inputs']['BLT1_theta_target'],
            search_space_data['Trajectory_Parameter_Inputs']['orbit_reserve_t'],
            rocket_param_data['Rocket_Parameters']['payload_mass_int'],
            search_space_data['Trajectory_Parameter_Inputs']['final_theta_target_int']
        ]
        
        self.n_ants = n_ants
        self.new_ants = new_ants
        self.n_cycles = n_cycles
        self.n_local_cycles = n_local_cycles
        self.elite_weight = elite_weight
        self.evaporation_rate = evaporation_rate
        self.nv = len(self.search_space)
        self.sigma = np.zeros(self.nv)
        self.archive_size = n_ants
        self.archive = []
        self.weights = np.zeros(self.archive_size)
        self.probability = np.zeros(self.archive_size)
        self.q = q
        self.elite_weight = elite_weight #Elitism Parameters
        self.elite_count = elite_count

    def evaluate_solution(self, solution):
        """Helper function to evaluate fitness."""
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = solution
        fitv, obj, fail_status = self.obj_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, flag = 0)       
        return obj, solution

    def initialize_archive(self):
        """Generate Initial Archive. """
        self.archive = []
        # Generate random solutions within bounds
        solutions = [
            [np.random.uniform(var[0], var[1]) for var in self.search_space]
            for _ in range(self.archive_size)
        ]
        
        # solutions = [
        #     [
        #         np.random.normal(
        #             loc=(var[0] + var[1]) / 2,  # Mean is the midpoint of bounds
        #             scale=(var[1] - var[0]) / 6  # Std dev is a fraction of the range (e.g., 1/4th)
        #         )
        #         for var in self.search_space
        #     ]
        #     for _ in range(self.archive_size)
        # ]
        
        # solutions = [
        #     [np.clip(value, var[0], var[1]) for value, var in zip(solution, self.search_space)]
        #     for solution in solutions
        # ]

        with mp.Pool() as pool:
            # Evaluate all solutions in parallel
            results = pool.map(self.evaluate_solution, solutions)
        self.archive = sorted(results, key=lambda x: x[0]) # Sort by objective value (ascending)

    def calculate_probability(self):
        """ Calculate probability of each solution in the archive."""
        q = self.q
        k = self.archive_size
        elite_count = self.elite_count
        elite_weight = self.elite_weight

        """Calculate Weight. """
        for i in range(1, self.archive_size+1):
            rank = i
            self.weights[i-1]= (np.exp(-((rank - 1)**2) / (2 * q**2 * k**2))) / (q * k * np.sqrt(2 * np.pi))
        
        """Calculate Probability. """
        for i in range(0, elite_count):
            self.probability[i] = elite_weight
        
        sum_weights = sum(self.weights[elite_count:])    
        for i in range(elite_count, self.archive_size):
            self.probability[i] = (self.weights[i]/sum_weights)*(1 - elite_count * elite_weight)
    
    def generate_new_solution(self):
        """Generate new solution"""
        k = self.archive_size
        m = self.new_ants
        nv = self.nv
        eta = self.evaporation_rate
        P = self.probability
        
        current_solution = np.array([sol[1] for sol in self.archive])
        
        roul = np.cumsum(P)

        all_new_solution = []
        for j in range(0, m):
            new_solution = []
            for n in range(0, nv):
                rand_num = random.random()
                choice = np.searchsorted(roul, rand_num) 
                """ Update the sigma"""
                sigma_vec = np.zeros(k)
                mean = current_solution[choice][n]
                for mm in range(k):
                    sigma_vec[mm]=abs(current_solution[choice][n]-current_solution[mm][n])
                sigma = eta*sum(sigma_vec)/(k-1)
                
                new_sol = np.random.normal(mean,sigma)
                ##clip the value between limit
                lim = self.search_space[n]
                if(new_sol<lim[0]):
                    new_sol = lim[0]
                if(new_sol>lim[1]):
                    new_sol = lim[1]
                new_solution.append(new_sol)
            all_new_solution.append(new_solution)

        with mp.Pool() as pool:
            new_archive = pool.map(self.evaluate_solution, all_new_solution)

        return new_archive
    
    def optimizer(self):
        print("======================================================")
        print("cycle    |     f_avg     |     fmin     ")
        print("======================================================")
        
        filename = f"current_archive_{formatted_time}.csv"
        self.initialize_archive()
        for cycle in range(self.n_cycles):
            print(f"{cycle+1 : < 8} |", end = " ", flush = True)
            self.calculate_probability()
            new_archive = self.generate_new_solution()
            combined_solutions = self.archive + new_archive
            combined_solutions.sort(key = lambda x: x[0])
            self.archive = combined_solutions[:self.archive_size]
            ##Save current archive
            save_archive_to_csv(self.archive, filename)
            
            f_min = self.archive[0][0]
            f_avg = np.mean([entry[0] for entry in self.archive])
            print(f"{f_avg: 13.10} | {f_min: 13.10}")

        best_archive = min(self.archive, key=lambda x: x[0])
        sigma_fraction = 0.10
        for i in range(self.nv):
            best_value = best_archive[1][i]  # The value of the best solution for dimension i
            sigma_i = sigma_fraction * abs(best_value)  # Use the absolute value of the best solution component
            self.sigma[i] = sigma_i
        
        """ Generate Rebal Ant"""
        for l_cycle in range(self.n_local_cycles):
            print(f'Local Cycle: {l_cycle}', flush = True)
            # self.initialize_archive()
            # for cycle in range(self.n_cycles):
            #     self.calculate_probability()
            #     new_archive = self.generate_new_solution()
            #     combined_solutions = self.archive + new_archive
            #     combined_solutions.sort(key = lambda x: x[0])
            #     self.archive = combined_solutions[:self.archive_size]
            #     save_archive_to_csv(self.archive, f"rebal_archive_{formatted_time}.csv")
            # best_rebal = min(self.archive, key=lambda x: x[0])

            # """Comparison current rebal solution with previous best"""
            # if(best_rebal[0]<best_archive[0]):
            #     best_archive = best_rebal
            
            self.archive = [best_archive]

            """ Generate k-1 local solution around the best solution by perturbing. """
            sigma = self.sigma
            all_solution = []
            for _ in range(self.archive_size-1):
                solution = []
                for i in range(self.nv):
                    UL = self.search_space[i][1]
                    LL = self.search_space[i][0]
                    var = best_archive[1][i] + np.random.normal(0, sigma[i])
                    var = np.clip(var, LL, UL)
                    solution.append(var)
                all_solution.append(solution)
            
            with mp.Pool() as pool:
                new_archive = pool.map(self.evaluate_solution, all_solution)

            self.archive = self.archive + new_archive
            self.archive.sort(key = lambda x: x[0]) #sort by fitness value
            save_archive_to_csv(self.archive, filename)

            """ Run Optimizer on new archive. """
            for cycle in range(self.n_cycles//16):
                print(f"{cycle+1 : < 8} |", end = " ", flush = True)
                self.calculate_probability()
                new_archive = self.generate_new_solution()
                combined_solutions = self.archive + new_archive
                combined_solutions.sort(key = lambda x: x[0])
                self.archive = combined_solutions[:self.archive_size]
                save_archive_to_csv(self.archive, filename)
                
                f_min = self.archive[0][0]
                f_avg = np.mean([entry[0] for entry in self.archive])
                print(f"{f_avg: 13.10} | {f_min: 13.10}")
            
            best_archive = min(self.archive, key=lambda x: x[0])
            self.sigma = self.sigma*(1-((l_cycle+1)/self.n_local_cycles))
            
        return best_archive
    
# Main script
import time
if __name__ == "__main__":
    start_time = time.time()
    search_space_json = "Input_JSON/optimize.json"
    run_simulation = Run_Simulation.Run_Simulation()  # Get the Simulation instance

    aco_r = ACOR(obj_function=run_simulation.run_open_callback, 
                 search_space_json=search_space_json, 
                 n_ants =20, 
                 new_ants = 5, 
                 n_cycles = 5000, 
                 n_local_cycles = 1,
                 q = 0.5, 
                 elite_count = 1, 
                 elite_weight = 0.2, 
                 evaporation_rate = 0.7, 
                 rocket_json = 'Input_JSON/vehicle.json')

    result = aco_r.optimizer()
    end_time = time.time()
    print(f'Minimum Value of the function: {result[0]} \n Best point: {result[1]}')
    print(f"Running Time: {(end_time - start_time)} sec.")
    

# aco_r = ACOR(obj_function=run_simulation.run_open_callback, search_space_json=search_space_json, n_ants =5, new_ants = 2, n_cycles = 2, n_local_cycles = 1)
# import inspect

# print("Method signature:", inspect.signature(Run_Simulation.Run_Simulation.run_open_callback))
# print("Method arguments:", Run_Simulation.Run_Simulation.run_open_callback.__code__.co_varnames)

# run_simulation = Run_Simulation.Run_Simulation()
# a, b, c = run_simulation.run_open_callback(2.26, 4.2, 0.781, 0.81, -0.807, 78368, 0, 285,263.3, 3.3, 66.2, 38.6, 173, 43.2, flag=0)
# print(a,b,c)

