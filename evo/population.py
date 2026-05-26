import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from evo.individual import Individual
from evo.utils import Setup, get_max_workers

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

# Global variables for worker processes to avoid redundant data serialization
_WORKER_DATA = None
_WORKER_LABELS = None

def _init_worker(data, labels):
    """Initializer for ProcessPoolExecutor workers"""
    global _WORKER_DATA, _WORKER_LABELS
    _WORKER_DATA = data
    _WORKER_LABELS = labels

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
            random_state=self.setup.RANDOM_SEED,
            penalty_factor=self.setup.PENALTY_FACTOR,
            cython_layout=self.setup.get_cython_layout(),
            enabled_models=[m for m in self.setup.INDIVIDUAL_CONFIG['models'] if m['enabled']],
            fitness_code=getattr(self.setup, 'FITNESS_CODE', None)
        )
        individual.fitness_eval(self.setup.DATA, self.setup.LABELS)
        return individual
    
    @staticmethod
    def _evaluate_individual(args):
        """Helper for ProcessPoolExecutor"""
        genes, filament_len, bits, project_folder, random_state, penalty_factor, DATA, LABELS, cython_layout, enabled_models, fitness_code = args
        from evo.individual import Individual
        ind = Individual(filament_len, genes, bits, project_folder, random_state, penalty_factor, cython_layout, enabled_models, fitness_code)
        
        # Use worker-local data if available to avoid serialization overhead
        global _WORKER_DATA, _WORKER_LABELS
        actual_data = DATA if DATA is not None else _WORKER_DATA
        actual_labels = LABELS if LABELS is not None else _WORKER_LABELS
        
        ind.fitness_eval(actual_data, actual_labels)
        # Return serializable results
        return {
            'fitness': ind.fitness,
            'acc': ind.acc,
            'f1': ind.f1,
            'prec': ind.prec,
            'recall': ind.recall,
            'cm': ind.cm,
            'model': ind.model,
            'preds': ind.preds,
            'genes': ind.genes
        }

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
        # Evaluate edge cases in main process (few enough)
        self._population = [self.init_individual(g) for g in edge_cases_genes]
        
        # 2. Initialize the rest of the population randomly
        remaining = self.setup.POP_SIZE - len(self._population)
        if remaining > 0:
            logger.info(f"Evaluating {remaining} individuals in parallel...")
            
            # Prepare arguments for ProcessPool with unique sub-seeds
            random_genes_list = []
            for _ in range(remaining):
                unpacked = self.setup.rng.choice(self.setup.GENES, size=self.setup.FILAMENT_LEN).astype(np.int8)
                random_genes_list.append(pack_bits(unpacked))
            
            # Use SeedSequence to generate independent child seeds for workers
            ss = np.random.SeedSequence(self.setup.RANDOM_SEED)
            child_seeds = ss.spawn(remaining)

            cython_layout = self.setup.get_cython_layout()
            enabled_models = [m for m in self.setup.INDIVIDUAL_CONFIG['models'] if m['enabled']]
            fitness_code = getattr(self.setup, 'FITNESS_CODE', None)

            eval_args = [
                (
                    genes, 
                    self.setup.FILAMENT_LEN, 
                    self.setup.BITS, 
                    self.setup.project_folder, 
                    int(child_seeds[i].generate_state(1)[0]), # Unique sub-seed
                    self.setup.PENALTY_FACTOR,
                    None, # Use worker-local data
                    None, # Use worker-local labels
                    cython_layout,
                    enabled_models,
                    fitness_code
                ) for i, genes in enumerate(random_genes_list)
            ]
            
            from pebble import ProcessPool
            from concurrent.futures import TimeoutError
            
            with ProcessPool(max_workers=get_max_workers(), initializer=_init_worker, initargs=(self.setup.DATA, self.setup.LABELS)) as executor:
                future = executor.map(Population._evaluate_individual, eval_args, timeout=self.setup.PATIENCE)
                
                iterator = future.result()
                results = []
                while True:
                    try:
                        res = next(iterator)
                        results.append(res)
                    except StopIteration:
                        break
                    except TimeoutError:
                        logger.warning("Individual evaluation timed out! Assigning penalty fitness.")
                        results.append({
                            'fitness': -1.0,
                            'acc': 0.0,
                            'f1': 0.0,
                            'prec': 0.0,
                            'recall': 0.0,
                            'cm': None,
                            'model': None,
                            'preds': None,
                            'genes': eval_args[len(results)][0]
                        })
                    except Exception as error:
                        logger.error(f"Individual evaluation failed: {error}")
                        results.append({
                            'fitness': -1.0,
                            'acc': 0.0,
                            'f1': 0.0,
                            'prec': 0.0,
                            'recall': 0.0,
                            'cm': None,
                            'model': None,
                            'preds': None,
                            'genes': eval_args[len(results)][0]
                        })
            
            for i, res in enumerate(results):
                ind = Individual(
                    self.setup.FILAMENT_LEN, 
                    res['genes'], 
                    self.setup.BITS, 
                    self.setup.project_folder, 
                    int(child_seeds[i].generate_state(1)[0]), # Use same sub-seed as evaluation
                    penalty_factor=self.setup.PENALTY_FACTOR,
                    cython_layout=cython_layout,
                    enabled_models=enabled_models,
                    fitness_code=fitness_code
                )
                ind._fitness = res['fitness']
                ind.acc = res['acc']
                ind.f1 = res['f1']
                ind.prec = res['prec']
                ind.recall = res['recall']
                ind.cm = res['cm']
                ind.model = res['model']
                ind.preds = res['preds']
                self._population.append(ind)
        
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
        cython_layout = self.setup.get_cython_layout()
        enabled_models = [m for m in self.setup.INDIVIDUAL_CONFIG['models'] if m['enabled']]
        fitness_code = getattr(self.setup, 'FITNESS_CODE', None)
        
        self._offspring = [
            Individual(
                self.setup.FILAMENT_LEN,
                genes=offspring_genes_pool[i],
                bits=self.setup.BITS,
                project_folder=self.setup.project_folder,
                random_state=self.setup.RANDOM_SEED,
                penalty_factor=self.setup.PENALTY_FACTOR,
                cython_layout=cython_layout,
                enabled_models=enabled_models,
                fitness_code=fitness_code
            ) for i in range(n_pop)
        ]
            
    def mutation(self, epoch: int, tot_epoch: int, alpha: float = 0.5):
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
        logger.info(f"Evaluating {len(mutated_pool)} mutated individuals in parallel...")
        
        # Use SeedSequence for independent child seeds
        ss = np.random.SeedSequence(self.setup.RANDOM_SEED + epoch) # Unique per generation
        child_seeds = ss.spawn(len(mutated_pool))

        cython_layout = self.setup.get_cython_layout()
        enabled_models = [m for m in self.setup.INDIVIDUAL_CONFIG['models'] if m['enabled']]
        fitness_code = getattr(self.setup, 'FITNESS_CODE', None)

        eval_args = [
            (
                mutated_pool[i], 
                self.setup.FILAMENT_LEN, 
                self.setup.BITS, 
                self.setup.project_folder, 
                int(child_seeds[i].generate_state(1)[0]), 
                self.setup.PENALTY_FACTOR,
                None, # Use worker-local data
                None, # Use worker-local labels
                cython_layout,
                enabled_models,
                fitness_code
            ) for i in range(len(mutated_pool))
        ]
        
        from pebble import ProcessPool
        from concurrent.futures import TimeoutError
        
        with ProcessPool(max_workers=get_max_workers(), initializer=_init_worker, initargs=(self.setup.DATA, self.setup.LABELS)) as executor:
            future = executor.map(Population._evaluate_individual, eval_args, timeout=self.setup.PATIENCE)
            
            iterator = future.result()
            results = []
            while True:
                try:
                    res = next(iterator)
                    results.append(res)
                except StopIteration:
                    break
                except TimeoutError:
                    logger.warning("Individual evaluation timed out! Assigning penalty fitness.")
                    results.append({
                        'fitness': -1.0,
                        'acc': 0.0,
                        'f1': 0.0,
                        'prec': 0.0,
                        'recall': 0.0,
                        'cm': None,
                        'model': None,
                        'preds': None,
                        'genes': eval_args[len(results)][0]
                    })
                except Exception as error:
                    logger.error(f"Individual evaluation failed: {error}")
                    results.append({
                        'fitness': -1.0,
                        'acc': 0.0,
                        'f1': 0.0,
                        'prec': 0.0,
                        'recall': 0.0,
                        'cm': None,
                        'model': None,
                        'preds': None,
                        'genes': eval_args[len(results)][0]
                    })
        
        self._offspring = []
        for i, res in enumerate(results):
            ind = Individual(
                self.setup.FILAMENT_LEN, 
                res['genes'], 
                self.setup.BITS, 
                self.setup.project_folder, 
                int(child_seeds[i].generate_state(1)[0]), # Use same sub-seed as evaluation
                penalty_factor=self.setup.PENALTY_FACTOR,
                cython_layout=cython_layout,
                enabled_models=enabled_models,
                fitness_code=fitness_code
            )
            ind._fitness = res['fitness']
            ind.acc = res['acc']
            ind.f1 = res['f1']
            ind.prec = res['prec']
            ind.recall = res['recall']
            ind.cm = res['cm']
            ind.model = res['model']
            ind.preds = res['preds']
            self._offspring.append(ind)

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
