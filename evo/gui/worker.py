import logging
import traceback
from PySide6.QtCore import QThread, Signal
from evo.runner import Runner
from evo.population import Population
from evo.utils import Setup

from main import preprocessing_general

class EvolutionWorker(QThread):
    generation_completed = Signal(int, float, float) # gen, best, avg
    finished = Signal(dict) # best results
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            # 1. Heavy Data Loading in Background (General)
            logging.info("📥 Loading and preprocessing data...")
            data_all, labels_all = preprocessing_general(
                self.params['train_path'], 
                self.params['val_path'], 
                self.params['test_path']
            )
            
            # For evolution, we use Train and Validation
            # If Val is None, we use Test as fallback for fitness eval
            X_train, X_val, X_test = data_all
            y_train, y_val, y_test = labels_all
            
            # If val is not provided, use a split of train or just use test if provided
            if X_val is None:
                if X_test is not None:
                    logging.info("⚠️ Validation set not provided, using Test set for evolution fitness.")
                    data_evo = (X_train, X_test)
                    labels_evo = (y_train, y_test)
                else:
                    # Fallback: simple split of train (e.g. 80/20) - for now just use train
                    logging.info("⚠️ Validation and Test sets not provided, using Train set for evolution fitness.")
                    data_evo = (X_train, X_train)
                    labels_evo = (y_train, y_train)
            else:
                data_evo = (X_train, X_val)
                labels_evo = (y_train, y_val)

            # 2. Setup Configuration
            setup = Setup(project_prefix='gui_exp_')
            setup.POP_SIZE = self.params['pop_size']
            setup.BITS = {
                'features': X_train.shape[1],
                'model_selection': 2,
                'model_params': 11,
            }
            setup.FILAMENT_LEN = sum(setup.BITS.values())
            setup.DATA = data_evo
            setup.LABELS = labels_evo
            setup.RANDOM_SEED = self.params.get('seed', 42)
            setup.seed_all(setup.RANDOM_SEED)
            setup.init_rng()

            # 3. Evolutionary Process
            pop = Population(setup=setup)
            runner = Runner(setup=setup, population=pop)
            
            logging.info('🐣 Initializing Population 🐤')
            pop.init_population()
            
            # Initial stats
            best_f = pop.best_individual.fitness
            avg_f = sum(ind.fitness for ind in pop.population) / len(pop.population)
            self.generation_completed.emit(0, best_f, avg_f)

            generations = self.params.get('generations', 10)
            alpha = self.params.get('alpha', 0.5)
            for epoch in range(generations):
                if not self.is_running:
                    logging.info('🛑 Evolution stopped by user.')
                    break
                
                runner.step(epoch, generations, alpha=alpha)
                
                best_f = pop.best_individual.fitness
                avg_f = sum(ind.fitness for ind in pop.population) / len(pop.population)
                self.generation_completed.emit(epoch + 1, best_f, avg_f)
                runner.log_top_five()

            if self.is_running:
                runner.log_tail()
                best = pop.best_individual
                results = {
                    'fitness': best.fitness,
                    'acc': best.acc,
                    'f1': best.f1,
                    'prec': best.prec,
                    'recall': best.recall,
                    'model_type': type(best.model).__name__,
                    'features_count': int(best.radiomics.sum()) if best.radiomics is not None else 0
                }
                self.finished.emit(results)

        except Exception as e:
            err_msg = traceback.format_exc()
            logging.error(f"FATAL: {err_msg}")
            self.error.emit(str(e))
