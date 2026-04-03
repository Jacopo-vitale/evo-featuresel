import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from evo.individual import Individual
from evo.utils import Setup

try:
    from evo.core import (
        fast_crossover_packed, 
        fast_mutation_packed, 
        pack_bits,
        batch_crossover_packed,
        batch_mutation_packed
    )
except ImportError:
    # Fallback (simplified)
    def pack_bits(x): return x
    def batch_crossover_packed(*args): return None
    def batch_mutation_packed(*args): return None

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

    def init_individual(self, genes=None):
        if genes is None or isinstance(genes, (int, np.integer)):
            # Generate random bits if no specific genes are provided
            unpacked_genes = self.setup.rng.choice(self.setup.GENES, size=self.setup.FILAMENT_LEN).astype(np.int8)
            genes = pack_bits(unpacked_genes)
        
        individual = Individual(
            self.setup.FILAMENT_LEN,
            genes=genes,
            bits=self.setup.BITS,
            project_folder=self.setup.project_folder,
            random_state=self.setup.RANDOM_SEED
        )
        individual.fitness_eval(self.setup.DATA, self.setup.LABELS)
        return individual
    
    def init_population(self):
        logger.info(f"Initializing population of size {self.setup.POP_SIZE}...")
        
        # 1. Create edge cases (unprobable individuals)
        edge_cases_genes = []
        
        # A. All bits set to 1
        all_ones = np.ones(self.setup.FILAMENT_LEN, dtype=np.int8)
        edge_cases_genes.append(pack_bits(all_ones))
        
        # B. Single-bit individuals (start, middle, end)
        for idx in [0, self.setup.FILAMENT_LEN // 2, self.setup.FILAMENT_LEN - 1]:
            single_bit = np.zeros(self.setup.FILAMENT_LEN, dtype=np.int8)
            single_bit[idx] = 1
            edge_cases_genes.append(pack_bits(single_bit))
            
        # C. A few random single-bit individuals
        for _ in range(3):
            random_idx = self.setup.rng.integers(0, self.setup.FILAMENT_LEN)
            single_bit = np.zeros(self.setup.FILAMENT_LEN, dtype=np.int8)
            single_bit[random_idx] = 1
            edge_cases_genes.append(pack_bits(single_bit))
            
        logger.info(f"Injecting {len(edge_cases_genes)} edge-case individuals (dense/sparse seeds)...")
        self._population = [self.init_individual(g) for g in edge_cases_genes]
        
        # 2. Initialize the rest of the population randomly
        remaining = self.setup.POP_SIZE - len(self._population)
        if remaining > 0:
            with ThreadPoolExecutor() as executor:
                self._population.extend(list(executor.map(self.init_individual, range(remaining))))
        
        self._population = sorted(self._population, key=lambda x: x.fitness, reverse=True)
        self.best_individual = self._population[0]
        
    def crossover(self):
        logger.info(f"Batch crossover using OpenMP...")
        
        # 1. Prepare indices and crossover points
        n_pop = self.setup.POP_SIZE
        half_pop = n_pop // 2
        
        p1_indices = self.setup.rng.integers(0, half_pop, size=n_pop).astype(np.int32)
        p2_indices = self.setup.rng.integers(half_pop, n_pop, size=n_pop).astype(np.int32)
        crossover_bits = self.setup.rng.integers(1, self.setup.FILAMENT_LEN - 1, size=n_pop).astype(np.int32)
        
        # 2. Extract genes into a 2D pool
        parents_pool = np.array([ind.genes for ind in self._population], dtype=np.uint8)
        
        # 3. Call C-level parallel crossover
        offspring_genes_pool = batch_crossover_packed(
            parents_pool,
            p1_indices,
            p2_indices,
            crossover_bits,
            self.setup.FILAMENT_LEN
        )
        
        # 4. Re-create Individual objects (this part is still Python but much faster than before)
        self._offspring = [
            Individual(
                self.setup.FILAMENT_LEN,
                genes=offspring_genes_pool[i],
                bits=self.setup.BITS,
                project_folder=self.setup.project_folder,
                random_state=self.setup.RANDOM_SEED
            ) for i in range(n_pop)
        ]
            
    def mutation(self, epoch: int, tot_epoch: int):
        alpha = 0.5
        self.mutation_rate = np.exp(-epoch / (tot_epoch * alpha))
        
        logger.info(f"Batch mutation using OpenMP (rate: {self.mutation_rate:.4f})...")
        
        # 1. Prepare 2D pool and random matrix
        offspring_pool = np.array([ind.genes for ind in self._offspring], dtype=np.uint8)
        random_matrix = self.setup.rng.uniform(0, 1, size=(len(self._offspring), self.setup.FILAMENT_LEN))
        
        # 2. Call C-level parallel mutation
        mutated_pool = batch_mutation_packed(
            offspring_pool,
            self.mutation_rate,
            random_matrix,
            self.setup.FILAMENT_LEN
        )
        
        # 3. Update Individual genes and re-evaluate fitness
        # We use ThreadPoolExecutor for fitness_eval as it's the main bottleneck (ML training)
        def evaluate(args):
            idx, genes = args
            ind = self._offspring[idx]
            ind.genes = genes
            ind.fitness_eval(self.setup.DATA, self.setup.LABELS)
            return ind

        with ThreadPoolExecutor() as executor:
            self._offspring = list(executor.map(evaluate, enumerate(mutated_pool)))

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