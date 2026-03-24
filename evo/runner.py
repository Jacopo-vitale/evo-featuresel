import logging
import sys
import os
from datetime import datetime
import numpy as np
from joblib import dump
from evo.population import Population
from evo.utils import Setup

# Setup module-level logger
logger = logging.getLogger("evo.runner")

class Runner(object):
    def __init__(self, setup: Setup = None, population: Population = None) -> None:
        self.setup = setup
        self.population = population
        
        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        # Use a consistent logger for the whole package
        main_logger = logging.getLogger("evo")
        main_logger.setLevel(logging.DEBUG)
        
        # Avoid adding handlers if they already exist
        if not main_logger.handlers:
            # Console handler
            stdout_handler = logging.StreamHandler(stream=sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(message)s")
            stdout_handler.setFormatter(fmt)
            main_logger.addHandler(stdout_handler)

            # File handler
            if self.setup and hasattr(self.setup, 'project_folder'):
                log_file = os.path.join(self.setup.project_folder, "experiment.log")
                os.makedirs(self.setup.project_folder, exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(fmt)
                main_logger.addHandler(file_handler)

    def step(self, epoch, generations):
        logger.info('👪 Starting Crossover 👪')
        self.population.crossover()
        logger.info(f'Mutant genes rate: {self.population.mutation_rate} ' + np.random.choice(['👽', '👾', '👹']))
        self.population.mutation(epoch, generations)
        logger.info('👴👵 Starting Replace 👦👧')
        self.population.replace()
        
    def run(self, generations: int = 10, target=1.0):
        self.welcome()
        if self.setup and self.setup.DESCRIPTION:
            self.description(self.setup.DESCRIPTION)
        
        logger.info('🐣 Initializing Population 🐤')
        logger.info('#' * 80)
        self.population.init_population()
    
        for epoch in range(generations):
            logger.info('*' * 80)
            logger.info(f'Starting epoch {epoch + 1}')
            self.step(epoch, generations)
            self.log_top_five()
            if self.population.bestindividual.fitness >= target:
                logger.info(f'Target achieved: {self.population.bestindividual.fitness}')
                logger.info('*' * 80)
                break
            
        logger.info('*' * 80)
        self.log_tail()
        
        if self.setup and hasattr(self.setup, 'project_folder'):
            result_path = os.path.join(self.setup.project_folder, 'iron_man.joblib')
            dump({
                'best_model': self.population.bestindividual.model,
                'best_genes': self.population.bestindividual.radiomics,
                'best_fitness': self.population.bestindividual.fitness,
                'best_acc': self.population.bestindividual.acc,
                'best_f1': self.population.bestindividual.f1,
                'best_recall': self.population.bestindividual.recall,
                'best_precision': self.population.bestindividual.prec,
                'best_cm': self.population.bestindividual.cm,
                'preds': self.population.bestindividual.preds
            }, result_path)
            logger.info(f"Best model saved to {result_path}")
    
    def welcome(self):
        # Cross-platform clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        welcome_message = """
###############################################################################
#                                                                             #
#                      ░▒▓█ WELCOME TO EVO-FEATURESEL █▓▒░                    #
#                                                                             #
#                     🔥🔥🔥        MULTITHREAD       🔥🔥🔥                   #
###############################################################################
        """
        logger.info(welcome_message)

    def description(self, descr):
        logger.info(descr)   
    
    def log_top_five(self):
        logger.info('-------- Top 5 Individuals Fitness --------')
        for i in range(min(5, len(self.population.population))):
            logger.info(f'I{i+1}: ' + f'{self.population.population[i].fitness:0.2f}')
    
    def log_tail(self):
        logger.info('--- Experiment summary ranking ---')
        # Log only top 10 if population is large to avoid clutter
        limit = min(10, len(self.population.population))
        for i in range(limit):
            ind = self.population.population[i]
            logger.info(f'Rank {i+1}: {ind.fitness:.4f} | Selected: {ind.genes.sum()} | Model: {type(ind.model).__name__ if ind.model else "None"}')

if __name__ == '__main__':
    # Basic smoke test or example could go here
    pass