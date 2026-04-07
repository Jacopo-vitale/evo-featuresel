import logging
import traceback
import numpy as np
from PySide6.QtCore import QThread, Signal
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedKFold
from evo.runner import Runner
from evo.population import Population
from evo.utils import Setup

from main import preprocessing_general

def run_single_fold(args):
    """
    Standalone function for running a single fold in a separate process.
    """
    fold_k, X_train, y_train, X_val, y_val, params = args
    
    # Configure Setup
    setup = Setup(project_prefix=f'gui_exp_fold_{fold_k}_')
    setup.POP_SIZE = params['pop_size']
    setup.BITS = {
        'features': X_train.shape[1],
        'model_selection': 2,
        'model_params': 11,
    }
    setup.FILAMENT_LEN = sum(setup.BITS.values())
    setup.DATA = (X_train, X_val)
    setup.LABELS = (y_train, y_val)
    setup.RANDOM_SEED = params.get('seed', 42) + fold_k # vary seed per fold
    setup.seed_all(setup.RANDOM_SEED)
    setup.init_rng()

    # To avoid nested ProcessPoolExecutor issues, we temporarily patch
    # the Population class to use ThreadPoolExecutor or sequential eval
    # However, since Python 3.7+ daemon processes can't have children.
    # We will rely on the fact that Population uses ProcessPoolExecutor,
    # which might fail if called from a ProcessPoolExecutor.
    # Alternatively, we just run sequentially in the QThread for outer CV
    # and use the inner ProcessPoolExecutor.

    # Let's run it. If it fails due to daemon process, we'll catch it.
    pop = Population(setup=setup)
    runner = Runner(setup=setup, population=pop)
    
    pop.init_population()
    
    generations = params.get('generations', 10)
    alpha = params.get('alpha', 0.5)
    
    for epoch in range(generations):
        runner.step(epoch, generations, alpha=alpha)
        
    best = pop.best_individual
    
    return {
        'fold': fold_k,
        'fitness': best.fitness,
        'acc': best.acc,
        'f1': best.f1,
        'prec': best.prec,
        'recall': best.recall,
        'model_type': type(best.model).__name__,
        'features_count': int(best.radiomics.sum()) if best.radiomics is not None else 0,
        'radiomics': best.radiomics
    }

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
            # 1. Data Loading
            logging.info("📥 Loading and preprocessing data...")
            data_all, labels_all = preprocessing_general(
                self.params['train_path'], 
                self.params['val_path'], 
                self.params['test_path'],
                self.params.get('train_labels_path'),
                self.params.get('val_labels_path'),
                self.params.get('test_labels_path')
            )
            
            X_train_full, X_val, X_test = data_all
            y_train_full, y_val, y_test = labels_all
            
            cv_folds = self.params.get('cv_folds', 1)
            
            if cv_folds > 1:
                logging.info(f"🔄 Starting {cv_folds}-fold Macro Cross-Validation...")
                
                # Combine train and val for CV
                if X_val is not None:
                    X_pool = np.vstack((X_train_full, X_val))
                    y_pool = np.concatenate((y_train_full, y_val))
                else:
                    X_pool = X_train_full
                    y_pool = y_train_full
                    
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.params.get('seed', 42))
                
                # We will run folds sequentially in the QThread to avoid nested ProcessPoolExecutor issues,
                # letting the Population class use its internal ProcessPoolExecutor for maximum speed
                # without OS thrashing.
                
                results_list = []
                for fold_k, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
                    if not self.is_running:
                        break
                    
                    logging.info(f"▶️ Starting Fold {fold_k + 1}/{cv_folds}")
                    X_tr, y_tr = X_pool[train_idx], y_pool[train_idx]
                    X_v, y_v = X_pool[val_idx], y_pool[val_idx]
                    
                    args = (fold_k, X_tr, y_tr, X_v, y_v, self.params)
                    
                    # Run fold sequentially to allow inner parallelization
                    res = run_single_fold(args)
                    results_list.append(res)
                    
                    # Update plot loosely
                    self.generation_completed.emit(fold_k + 1, res['fitness'], res['fitness'])

                if not self.is_running:
                    return

                # Aggregate results
                avg_fitness = np.mean([r['fitness'] for r in results_list])
                std_fitness = np.std([r['fitness'] for r in results_list])
                avg_acc = np.mean([r['acc'] for r in results_list])
                avg_f1 = np.mean([r['f1'] for r in results_list])
                avg_prec = np.mean([r['prec'] for r in results_list])
                avg_recall = np.mean([r['recall'] for r in results_list])
                avg_features = np.mean([r['features_count'] for r in results_list])
                
                # Most common model
                models = [r['model_type'] for r in results_list]
                most_common_model = max(set(models), key=models.count)
                
                final_results = {
                    'fitness': avg_fitness,
                    'acc': avg_acc,
                    'f1': avg_f1,
                    'prec': avg_prec,
                    'recall': avg_recall,
                    'model_type': f"{most_common_model} (CV mode)",
                    'features_count': avg_features,
                    'std_fitness': std_fitness
                }
                
                logging.info(f"🏆 CV Completed. Avg MCC: {avg_fitness:.4f} ± {std_fitness:.4f}")
                self.finished.emit(final_results)

            else:
                # Normal single run
                if X_val is None:
                    if X_test is not None:
                        data_evo = (X_train_full, X_test)
                        labels_evo = (y_train_full, y_test)
                    else:
                        data_evo = (X_train_full, X_train_full)
                        labels_evo = (y_train_full, y_train_full)
                else:
                    data_evo = (X_train_full, X_val)
                    labels_evo = (y_train_full, y_val)

                setup = Setup(project_prefix='gui_exp_')
                setup.POP_SIZE = self.params['pop_size']
                setup.BITS = {
                    'features': X_train_full.shape[1],
                    'model_selection': 2,
                    'model_params': 11,
                }
                setup.FILAMENT_LEN = sum(setup.BITS.values())
                setup.DATA = data_evo
                setup.LABELS = labels_evo
                setup.RANDOM_SEED = self.params.get('seed', 42)
                setup.seed_all(setup.RANDOM_SEED)
                setup.init_rng()

                pop = Population(setup=setup)
                runner = Runner(setup=setup, population=pop)
                
                logging.info('🐣 Initializing Population 🐤')
                pop.init_population()
                
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
