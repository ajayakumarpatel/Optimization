"""
Code Developed By: Ajaya Kumar Patel
Version: 1.0
Description: Ant Colony Optimizer
Release Date: 03-12-2024
"""

import numpy as np
import random

class ACOR:
    def __init__(self, obj_function, search_space, n_ants = 5, new_ants = 2, n_cycles = 400, n_local_cycles = 5, q = 0.5, elite_count = 1, elite_weight = 0.4, evaporation_rate = 0.7, sigma = 0.5):
        self.obj_function = obj_function
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
        for _ in range(self.archive_size):
            solution = []
            for i in range(self.nv):
                var = self.search_space[i]
                solution.append(np.random.uniform(var[0], var[1]))  # Generate a random value within bounds
            # print(*solution)
            fitness = self.obj_function(*solution)
            self.archive.append((fitness, solution))
        self.archive.sort(key=lambda x: x[0])  # Sort by objective value (ascending)

    def calculate_probability(self):
        """ Calculate probability of each solution in the archive."""
        q = self.q
        k = self.archive_size
        elite_count = self.elite_count
        elite_weight = self.elite_weight
        ##Calculate Weight
        for i in range(1, self.archive_size+1):
            rank = i
            self.weights[i-1]= (np.exp(-((rank - 1)**2) / (2 * q**2 * k**2))) / (q * k * np.sqrt(2 * np.pi))
        
        ##Calculate Probability
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
        new_archive = []
        current_fitness = np.array([sol[0] for sol in self.archive])
        current_solution = np.array([sol[1] for sol in self.archive])
        
        roul = np.zeros(k)
        roul[0] = P[0]
        for i in range(1, k):
            roul[i] = roul[i-1] + P[i]

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
            
            fitness = self.obj_function(*new_solution)
            new_archive.append((fitness, new_solution))
        return new_archive
    
    def optimizer(self):
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
            for _ in range(self.archive_size-1):
                solution = []
                for i in range(self.nv):
                    UL = self.search_space[i][1]
                    LL = self.search_space[i][0]
                    var = best_archive[1][i] + np.random.normal(0, sigma) * (UL - LL)
                    var = np.clip(var, LL, UL)
                    solution.append(var)
                fitness = self.obj_function(*solution)
                self.archive.append((fitness, solution))
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