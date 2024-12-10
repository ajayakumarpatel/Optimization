"""
Code Title: Modified Ant Colony Optimization (ACOR) Algorithm
Version: V2.0
Developed By: Ajaya Kumar Patel
Release Date: 10-12-2024

Description:
This script implements a modified version of Ant Colony Optimization (ACOR) algorithm for solving optimization problems. 
The algorithm is inspired by the behavior of ants in finding optimal paths.

Features:
- Parallel evaluation of solutions using GPU for faster computation.
- Customizable search space and objective function.

Contact:
For feedback or inquiries, please contact: [ajayakumarpatel@agnikul.in]
"""


import numpy as np
import random
from numba import cuda
from obj_func import run_simulation


@cuda.jit
def evaluate_solutions(solutions, fitnesses):
    """GPU evaluate solutions."""
    idx = cuda.grid(1)  # Get thread index
    if idx < solutions.shape[0]:  # Ensure thread index is within bounds
        x, y = solutions[idx]
        fitnesses[idx] = run_simulation(x, y)

class ACOR:
    def __init__(self, search_space, n_ants = 20, new_ants = 2, n_cycles = 400, n_local_cycles = 5, q = 0.5, elite_count = 1, elite_weight = 0.4, evaporation_rate = 0.7, sigma = 0.5):
        self.search_space = search_space
        self.n_ants = n_ants
        self.new_ants = new_ants
        self.n_cycles = n_cycles
        self.n_local_cycles = n_local_cycles
        self.elite_weight = elite_weight
        self.evaporation_rate = evaporation_rate
        self.sigma = sigma
        self.nv = len(search_space)
        self.archive_size = n_ants
        self.archive = []
        self.weights = np.zeros(self.archive_size)
        self.probability = np.zeros(self.archive_size)
        self.q = q
        self.elite_weight = elite_weight #Elitism Parameters
        self.elite_count = elite_count

    def initialize_archive(self):
        """Generate Initial Archive. """
        self.archive = []
        # Generate random solutions within bounds
        solutions = [
            [np.random.uniform(var[0], var[1]) for var in self.search_space]
            for _ in range(self.archive_size)
        ]

        # Allocate arrays on GPU
        d_solutions = cuda.to_device(solutions)
        d_fitnesses = cuda.device_array(self.archive_size)

        # Define GPU thread and block configuration
        threads_per_block = 32
        blocks_per_grid = (self.archive_size + (threads_per_block - 1)) // threads_per_block        
        #Launch in GPU
        evaluate_solutions[blocks_per_grid, threads_per_block](d_solutions, d_fitnesses)

        fitnesses = d_fitnesses.copy_to_host()
        self.archive = sorted(zip(fitnesses, solutions), key=lambda x: x[0])

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
        
        roul = np.zeros(k)
        roul[0] = P[0]
        for i in range(1, k):
            roul[i] = roul[i-1] + P[i]

        all_new_solution = []
        for j in range(0, m):
            new_solution = []
            for n in range(0, nv):
                rand_num = random.random()
                count = 0
                choice = 0
                while(choice==0):
                    if(rand_num<=roul[count]):
                        choice = count
                    count = count + 1
                
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
        
        d_solutions = cuda.to_device(all_new_solution)
        d_fitnesses = cuda.device_array(m)
        threads_per_block = 32
        blocks_per_grid = (self.archive_size + (threads_per_block - 1)) // threads_per_block
        evaluate_solutions[blocks_per_grid, threads_per_block](d_solutions, d_fitnesses)
        fitnesses = d_fitnesses.copy_to_host()
        new_archive = list(zip(fitnesses, all_new_solution))

        return new_archive
    
    def run(self):        
        self.initialize_archive()
        for cycle in range(self.n_cycles):
            # print(cycle, end = ' ', flush = True)
            self.calculate_probability()
            new_archive = self.generate_new_solution()
            combined_solutions = self.archive + new_archive
            combined_solutions.sort(key = lambda x: x[0])
            self.archive = combined_solutions[:self.archive_size]

        best_archive = min(self.archive, key=lambda x: x[0])
        
        """ Generate Rebal Ant"""
        for l_cycle in range(self.n_local_cycles):
            # print(f'Local Cycle: {l_cycle}', flush = True)
            self.initialize_archive()
            for cycle in range(self.n_cycles):
                self.calculate_probability()
                new_archive = self.generate_new_solution()
                combined_solutions = self.archive + new_archive
                combined_solutions.sort(key = lambda x: x[0])
                self.archive = combined_solutions[:self.archive_size]
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
            
            d_solutions = cuda.to_device(all_solution)
            d_fitnesses = cuda.device_array(self.archive_size-1)
            threads_per_block = 32
            blocks_per_grid = (self.archive_size + (threads_per_block - 1)) // threads_per_block
            evaluate_solutions[blocks_per_grid, threads_per_block](d_solutions, d_fitnesses)
            fitnesses = d_fitnesses.copy_to_host()
            new_archive = list(zip(fitnesses, all_solution))
            self.archive = self.archive + new_archive
            self.archive.sort(key = lambda x: x[0]) #sort by fitness value

            """ Run Optimizer on new archive. """
            for cycle in range(self.n_cycles):
                self.calculate_probability()
                new_archive = self.generate_new_solution()
                combined_solutions = self.archive + new_archive
                combined_solutions.sort(key = lambda x: x[0])
                self.archive = combined_solutions[:self.archive_size]
            best_archive = min(self.archive, key=lambda x: x[0])
            self.sigma = self.sigma*(1-((l_cycle+1)/self.n_local_cycles))
            
        return best_archive