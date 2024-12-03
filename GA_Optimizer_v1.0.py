"""
Author: Ajaya Kumar Patel
Version: 1.0
Description: Genetic Algorithm Optimizer
Release Date: 03-12-2024
"""

import numpy as np
import random

class GA_Optimization:
    def __init__(self, obj_function, search_space, population_size, generations=500, maximize=True, crossover_weight = 1/3, crossover_rate=0.8, mutation_rate=0.15, mut_alpha = 0.1):
        self.obj_function = obj_function
        self.search_space = search_space
        self.population_size = population_size
        self.chromosome_length = len(search_space)
        self.maximize = maximize
        self.mutation_rate = mutation_rate
        self.mut_alpha = mut_alpha
        self.crossover_weight = crossover_weight
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = np.zeros((population_size, len(search_space)))
        self.fitness_score = np.zeros(population_size)
        
    def initialize_population(self):
        """Generate Initial Population."""
        for i in range(self.population_size):
            solution = []
            for j in range(self.chromosome_length):
                lim = self.search_space[j]
                solution.append(np.random.uniform(lim[0], lim[1]))
            self.population[i] = solution
    
    def evaluate_population(self):
        """Determine finess of the population."""
        for i in range(self.population_size):
            sol = self.population[i]
            fitness = self.obj_function(*sol)
            if self.maximize == False:
                fitness = 1/fitness 
            self.fitness_score[i] = fitness

    def select_parents_roulette(self):
        """Select parents using roulette. """
        total_fitness = sum(self.fitness_score)
        probabilities = [fitness / total_fitness for fitness in self.fitness_score]
        parents = random.choices(self.population, probabilities, k = 2)
        return parents

    def crossover(self, parent1, parent2):
        """Perform arithmetics crossover."""
        if random.random() < self.crossover_rate:
            weight = self.crossover_weight
            child1 = weight * parent1 + (1 - weight) * parent2
            child2 = weight * parent2 + (1 - weight) * parent1
        else:
            child1, child2 = parent1, parent2
        return child1, child2

    def mutate(self, chromosome):
        """Perform Gaussian Mutation."""
        for i in range(self.chromosome_length):
            if(random.random()<self.mutation_rate):
                lim = self.search_space[i]
                sigma = abs(chromosome[i]*self.mut_alpha)
                mutated_chromosome = np.random.normal(chromosome[i], sigma)
                chromosome[i] = min(max(mutated_chromosome,lim[0]), lim[1])
        return chromosome

    def evolve(self):
        """Run the evolutionary process."""
        # Find the best chromosome(s) in the current population
        elite_count = 2  # Number of elite individuals to preserve
        elite_indices = sorted(range(len(self.fitness_score)), key=lambda i: self.fitness_score[i], reverse=True)[:elite_count]
        elites = [self.population[i] for i in elite_indices]
    
        # Generate the new population
        new_population = []
        while len(new_population) < self.population_size - elite_count:
            parent1, parent2 = self.select_parents_roulette()
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            if len(new_population) < self.population_size - elite_count:
                new_population.append(self.mutate(child2))
        # Add elites to the new population
        self.population = new_population + elites
    
    def optimizer(self):
        """Execute the GA for the specified number of generations."""
        self.initialize_population()
        # print(self.population)
        for generation in range(self.generations):
            self.evaluate_population()
            best_fitness_idx = np.argmax(self.fitness_score)
            best_fitness = np.max(self.fitness_score) if self.maximize else (1/np.max(self.fitness_score))
            best_individual = self.population[best_fitness_idx]
            # print(f"Generation {generation + 1}: Best solution = {best_individual}, fitness = {best_fitness}")
            self.evolve()
            
        return best_individual, best_fitness
    
if __name__ == "__main__": 
    def camelback(x, y):
        return (4 - 2.1 * x**2 + (x**4) / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2
    
    def ackley(x, y):
        term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        return term1 + term2 + 20 + np.e
    
    search_space = [(-5, 5), (-5, 5)]
    GAO = GA_Optimization(obj_function = ackley, search_space = search_space, population_size = 50, generations=500, maximize=False)
    best_individual, best_fitness = GAO.optimizer()
    print(f' Minimum Value of the function: {best_fitness} \n Best point: {best_individual}')