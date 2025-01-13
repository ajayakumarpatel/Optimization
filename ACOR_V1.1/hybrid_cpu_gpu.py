import numpy as np
import random
import cupy as cp
import multiprocessing


random.seed(0)
class ACOR:
    def __init__(self, obj_function, search_space, n_ants=20, new_ants=2, n_cycles=400, n_local_cycles=5, q=0.5, elite_count=1, elite_weight=0.4, evaporation_rate=0.7, sigma=0.5):
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
        self.weights = cp.zeros(self.archive_size)  # GPU array
        self.probability = cp.zeros(self.archive_size)  # GPU array
        self.q = q
        self.elite_weight = elite_weight  # Elitism Parameters
        self.elite_count = elite_count


    def evaluate_solution(self, solution):
        """Helper function to evaluate fitness on CPU."""
        fitness = self.obj_function(*solution)        
        return fitness, solution

    def initialize_archive(self):
        """Generate Initial Archive."""
        self.archive = []
        # Generate random solutions within bounds
        solutions = [
            [np.random.uniform(var[0], var[1]) for var in self.search_space]
            for _ in range(self.archive_size)
        ]
        
        with multiprocessing.Pool(processes=8) as pool:
            results = pool.map(self.evaluate_solution, solutions)

        self.archive = sorted(results, key=lambda x: x[0])  # Sort by objective value (ascending)

    def calculate_probability(self):
        """ Calculate probability of each solution in the archive."""
        q = self.q
        k = self.archive_size
        elite_count = self.elite_count
        elite_weight = self.elite_weight

        """Calculate Weight. """
        ranks = cp.arange(1, k + 1)  # Generate ranks directly on GPU
        self.weights = cp.exp(-((ranks - 1) ** 2) / (2 * q ** 2 * k ** 2)) / (q * k * cp.sqrt(2 * cp.pi))
        
        """Calculate Probability. """
        self.probability[:elite_count] = elite_weight
        
        sum_weights = cp.sum(self.weights[elite_count:])    
        self.probability[elite_count:] = (self.weights[elite_count:] / sum_weights) * (1 - elite_count * elite_weight)
            
    def generate_new_solution(self):
        """Generate new solution"""
        k = self.archive_size
        m = self.new_ants
        nv = self.nv
        eta = self.evaporation_rate
        P = self.probability

        current_solution = cp.array([sol[1] for sol in self.archive])  # GPU array
        roul = cp.cumsum(P)

        all_new_solution = []
        for j in range(0, m):
            new_solution = []
            for n in range(0, nv):
                rand_nums = cp.random.rand(1)  # Generate one random number
                choice = cp.searchsorted(roul, rand_nums) 
                
                """ Update the sigma"""
                mean = current_solution[choice, n]
                sigma_vec = cp.abs(current_solution[:, n] - mean)
                sigma = eta * cp.sum(sigma_vec) / (k - 1)
                
                new_sol = np.random.normal(mean.item(), sigma.item()) 
                new_sol = np.clip(new_sol, self.search_space[n][0], self.search_space[n][1])  # Clip to limits
                new_solution.append(new_sol)
            all_new_solution.append(new_solution)

        with multiprocessing.Pool(processes=8) as pool:
            new_archive = pool.map(self.evaluate_solution, all_new_solution)
        return new_archive

    
    def optimizer(self):
        self.initialize_archive()
        for cycle in range(self.n_cycles):
            print(cycle, end =' ', flush = True)
            self.calculate_probability()
            new_archive = self.generate_new_solution()
            combined_solutions = self.archive + new_archive
            combined_solutions.sort(key=lambda x: x[0])
            self.archive = combined_solutions[:self.archive_size]

        best_archive = min(self.archive, key=lambda x: x[0])
        
        """ Generate Rebal Ant"""
        for l_cycle in range(self.n_local_cycles):
            print(l_cycle, end =' ', flush = True)
            self.initialize_archive()
            for cycle in range(self.n_cycles):
                self.calculate_probability()
                new_archive = self.generate_new_solution()
                combined_solutions = self.archive + new_archive
                combined_solutions.sort(key=lambda x: x[0])
                self.archive = combined_solutions[:self.archive_size]
            best_rebal = min(self.archive, key=lambda x: x[0])

            """ Comparison current rebal solution with previous best"""
            if best_rebal[0] < best_archive[0]:
                best_archive = best_rebal
            self.archive = [best_archive]

            """ Generate k-1 local solution around the best solution by perturbing."""
            sigma = self.sigma
            all_solution = []
            for _ in range(self.archive_size - 1):
                solution = []
                for i in range(self.nv):
                    UL = self.search_space[i][1]
                    LL = self.search_space[i][0]
                    var = best_archive[1][i] + np.random.normal(0, sigma) * (UL - LL)
                    solution.append(np.clip(var, LL, UL))

                all_solution.append(solution)
            
            with multiprocessing.Pool(processes=8) as pool:
                new_archive = pool.map(self.evaluate_solution, all_solution)
                        
            self.archive = self.archive + new_archive
            self.archive.sort(key=lambda x: x[0])  # Sort by fitness value

            """ Run Optimizer on new archive. """
            for cycle in range(self.n_cycles):
                print(cycle, end =' ', flush = True)
                self.calculate_probability()
                new_archive = self.generate_new_solution()
                combined_solutions = self.archive + new_archive
                combined_solutions.sort(key=lambda x: x[0])
                self.archive = combined_solutions[:self.archive_size]
            best_archive = min(self.archive, key=lambda x: x[0])
            self.sigma = self.sigma * (1 - ((l_cycle + 1) / self.n_local_cycles))
            
        return best_archive
