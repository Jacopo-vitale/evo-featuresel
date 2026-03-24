import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from evo.individual import Individual
from evo.utils import Setup

# Setup module-level logger
logger = logging.getLogger("evo.population")

class Population(object):
    def __init__(self, setup: Setup) -> None:
        self.setup = setup
        self._population: list[Individual] = []
        self._offspring: list[Individual] = []
        self.mutation_rate = 1.0
        self.best_individual = None
        
    @property
    def population(self):
        return self._population
    
    @property
    def offspring(self):
        return self._offspring

    @property
    def bestindividual(self):
        # Compatibility with existing runner
        return self.best_individual

    def init_individual(self, _=None):
        individual = Individual(
            self.setup.FILAMENT_LEN,
            genes=self.setup.rng.choice(self.setup.GENES, size=self.setup.FILAMENT_LEN),
            bits=self.setup.BITS,
            project_folder=self.setup.project_folder,
            random_state=self.setup.RANDOM_SEED
        )
        individual.fitness_eval(self.setup.DATA, self.setup.LABELS)
        return individual
    
    def init_population(self):
        logger.info(f"Initializing population of size {self.setup.POP_SIZE}...")
        with ThreadPoolExecutor() as executor:
            self._population = list(executor.map(self.init_individual, range(self.setup.POP_SIZE)))
        
        self._population = sorted(self._population, key=lambda x: x.fitness, reverse=True)
        self.best_individual = self._population[0]
        
    def crossover_individuals(self, _=None):
        crossover_point = self.setup.rng.integers(1, self.setup.FILAMENT_LEN - 1)
        
        # Select parents: one from top 50%, one from bottom 50% (as in original code)
        half_pop = int(0.5 * self.setup.POP_SIZE)
        parent1 = self.setup.rng.choice(self._population[:half_pop])
        parent2 = self.setup.rng.choice(self._population[half_pop:])
        
        child_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])
        
        return Individual(
            filament_len=self.setup.FILAMENT_LEN,
            genes=child_genes,
            bits=self.setup.BITS,
            project_folder=self.setup.project_folder,
            random_state=self.setup.RANDOM_SEED
        )
        
    def crossover(self):
        with ThreadPoolExecutor() as executor:
            self._offspring = list(executor.map(self.crossover_individuals, range(self.setup.POP_SIZE)))
            
    def create_mutant(self, individual):
        # We need to work on a copy to avoid mutating the original genes if they were reused
        mutated_genes = individual.genes.copy()
        mutation_mask = self.setup.rng.uniform(0, 1, size=len(mutated_genes)) < self.mutation_rate
        mutated_genes[mutation_mask] = 1 - mutated_genes[mutation_mask]
        
        # Update genes and re-evaluate fitness
        individual.genes = mutated_genes
        individual.fitness_eval(self.setup.DATA, self.setup.LABELS)
        return individual
    
    def mutation(self, epoch: int, tot_epoch: int):
        alpha = 0.5
        self.mutation_rate = np.exp(-epoch / (tot_epoch * alpha))
        
        with ThreadPoolExecutor() as executor:
            self._offspring = list(executor.map(self.create_mutant, self._offspring))        

        self._offspring = sorted(self._offspring, key=lambda x: x.fitness, reverse=True)

    def replace(self):
        tmp_generation = []
        for old, new in zip(self._population, self._offspring):
            if old.fitness < new.fitness:
                tmp_generation.append(new)
            else:
                tmp_generation.append(old)
            
        self._population = sorted(tmp_generation, key=lambda x: x.fitness, reverse=True)
        self._offspring.clear()
        self.best_individual = self._population[0]