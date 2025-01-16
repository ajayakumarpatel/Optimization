"""
Code Title: Modified Ant Colony Optimization (ACOR) Algorithm
Version: 1.0
Developed By: Ajaya Kumar Patel
Release Date: 03-12-2024

Description:
This script implements a modified version of Ant Colony Optimization (ACOR) algorithm for solving optimization problems. 
The algorithm is inspired by the behavior of ants in finding optimal paths.

Features:
- Parallel evaluation of solutions using multi thread.
- Customizable search space and objective function.

Contact:
For feedback or inquiries, please contact: [ajayakumarpatel@agnikul.in]
"""
import json
import numpy as np
import random
import multiprocessing
import csv
import time
from datetime import datetime

random.seed(0)
time_stamp = time.time()
formatted_time = datetime.fromtimestamp(time_stamp).strftime('%H%M%S_%d%m%y')

def save_archive_to_csv(archive, filename):
    with open(filename, mode = 'w', newline='') as file:
        writer = csv.writer(file)
        header = ["Fitness"] + [f"Variable_{i}" for i in range(1, len(archive[0][1]) + 1)]
        writer.writerow(header)

        for solution in archive:
            fitness = solution[0]
            variables = solution[1]
            row = [fitness] + variables
            writer.writerow(row)


class ACOR:
    def __init__(self, obj_function, search_space_json, n_ants = 20, new_ants = 2, n_cycles = 400, n_local_cycles = 5, q = 0.5, elite_count = 1, elite_weight = 0.4, evaporation_rate = 0.7, sigma = 0.5):
        with open(search_space_json, 'r') as file:
            search_space_dict = json.load(file)
        
        self.obj_function = obj_function
        # self.search_space = search_space
        self.search_space = [
            (param[0], param[1]) for param in search_space_dict['parameters'].values()
        ]
        self.n_ants = n_ants
        self.new_ants = new_ants
        self.n_cycles = n_cycles
        self.n_local_cycles = n_local_cycles
        self.elite_weight = elite_weight
        self.evaporation_rate = evaporation_rate
        self.sigma = sigma
        self.nv = len(self.search_space)
        self.archive_size = n_ants
        self.archive = []
        self.weights = np.zeros(self.archive_size)
        self.probability = np.zeros(self.archive_size)
        self.q = q
        self.elite_weight = elite_weight #Elitism Parameters
        self.elite_count = elite_count

    def evaluate_solution(self, solution):
        """Helper function to evaluate fitness."""
        fitness = self.obj_function(*solution)        
        return fitness, solution

    def initialize_archive(self):
        """Generate Initial Archive. """
        self.archive = []
        # Generate random solutions within bounds
        solutions = [
            [np.random.uniform(var[0], var[1]) for var in self.search_space]
            for _ in range(self.archive_size)
        ]
        
        with multiprocessing.Pool(processes=8) as pool:
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
        current_fitness = np.array([sol[0] for sol in self.archive])
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

        with multiprocessing.Pool(processes=m) as pool:
            new_archive = pool.map(self.evaluate_solution, all_new_solution)

        return new_archive
    
    def optimizer(self):
        filename = f"current_archive_{formatted_time}.csv"
        self.initialize_archive()
        for cycle in range(self.n_cycles):
            print(cycle, end = ' ', flush = True)
            self.calculate_probability()
            new_archive = self.generate_new_solution()
            combined_solutions = self.archive + new_archive
            combined_solutions.sort(key = lambda x: x[0])
            self.archive = combined_solutions[:self.archive_size]
            ##Save current archive
            save_archive_to_csv(self.archive, filename)

        best_archive = min(self.archive, key=lambda x: x[0])
        
        """ Generate Rebal Ant"""
        for l_cycle in range(self.n_local_cycles):
            print(f'Local Cycle: {l_cycle}', flush = True)
            self.initialize_archive()
            for cycle in range(self.n_cycles):
                self.calculate_probability()
                new_archive = self.generate_new_solution()
                combined_solutions = self.archive + new_archive
                combined_solutions.sort(key = lambda x: x[0])
                self.archive = combined_solutions[:self.archive_size]
                save_archive_to_csv(self.archive, f"rebal_archive_{formatted_time}.csv")
            best_rebal = min(self.archive, key=lambda x: x[0])

            """Comparison current rebal solution with previous best"""
            if(best_rebal[0]<best_archive[0]):
                best_archive = best_rebal
            self.archive = [best_archive]

            """ Generate k-1 local solution around the best solution by perturbing. """
            sigma = self.sigma
            all_solution = []
            for _ in range(self.archive_size-1):
                solution = []
                for i in range(self.nv):
                    UL = self.search_space[i][1]
                    LL = self.search_space[i][0]
                    var = best_archive[1][i] + np.random.normal(0, sigma) * (UL - LL)
                    var = np.clip(var, LL, UL)
                    solution.append(var)
                all_solution.append(solution)
            
            with multiprocessing.Pool(processes=8) as pool:
                new_archive = pool.map(self.evaluate_solution, all_solution)

            self.archive = self.archive + new_archive
            self.archive.sort(key = lambda x: x[0]) #sort by fitness value
            save_archive_to_csv(self.archive, filename)

            """ Run Optimizer on new archive. """
            for cycle in range(self.n_cycles):
                # print(cycle, end = ' ', flush = True)
                self.calculate_probability()
                new_archive = self.generate_new_solution()
                combined_solutions = self.archive + new_archive
                combined_solutions.sort(key = lambda x: x[0])
                self.archive = combined_solutions[:self.archive_size]
                save_archive_to_csv(self.archive, filename)
            best_archive = min(self.archive, key=lambda x: x[0])
            self.sigma = self.sigma*(1-((l_cycle+1)/self.n_local_cycles))
            
        return best_archive
