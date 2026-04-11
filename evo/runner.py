import logging
import sys
import os
import csv
import json
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
        fmt = logging.Formatter("%(message)s")

        if not any(
            isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout
            for handler in main_logger.handlers
        ):
            stdout_handler = logging.StreamHandler(stream=sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(fmt)
            main_logger.addHandler(stdout_handler)

        if self.setup and hasattr(self.setup, 'project_folder'):
            log_file = os.path.abspath(os.path.join(self.setup.project_folder, "experiment.log"))
            os.makedirs(self.setup.project_folder, exist_ok=True)

            stale_handlers = [
                handler for handler in main_logger.handlers
                if isinstance(handler, logging.FileHandler)
                and os.path.abspath(getattr(handler, "baseFilename", "")) != log_file
            ]
            for handler in stale_handlers:
                main_logger.removeHandler(handler)
                handler.close()

            has_file_handler = any(
                isinstance(handler, logging.FileHandler)
                and os.path.abspath(getattr(handler, "baseFilename", "")) == log_file
                for handler in main_logger.handlers
            )
            if not has_file_handler:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(fmt)
                main_logger.addHandler(file_handler)

    def save_best_result(self):
        if not (self.setup and hasattr(self.setup, 'project_folder') and self.population and self.population.bestindividual):
            return None

        result_path = os.path.abspath(os.path.join(self.setup.project_folder, 'iron_man.joblib'))
        os.makedirs(self.setup.project_folder, exist_ok=True)
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
        return result_path

    def step(self, epoch, generations, alpha: float = 0.5):
        logger.info('👪 Starting Crossover 👪')
        self.population.crossover()
        logger.info(f'Mutant genes rate: {self.population.mutation_rate} ' + np.random.choice(['👽', '👾', '👹']))
        self.population.mutation(epoch, generations, alpha=alpha)
        logger.info('👴👵 Starting Replace 👦👧')
        self.population.replace()
        
    def run(self, generations: int = 10, target=1.0, alpha: float = 0.5):
        self.welcome()
        if self.setup and self.setup.DESCRIPTION:
            self.description(self.setup.DESCRIPTION)
        
        n_runs = getattr(self.setup, 'N_ROBUSTNESS_RUNS', 1)
        best_of_runs = []

        for run_idx in range(n_runs):
            if n_runs > 1:
                logger.info(f"\n🚀 STARTING ROBUSTNESS RUN {run_idx + 1}/{n_runs} 🚀")
                # Reseed for this specific run
                self.setup.seed_all(self.setup.RANDOM_SEED + run_idx)
                # Re-initialize population for new seed
                self.population.init_population()
            else:
                logger.info('🐣 Initializing Population 🐤')
                logger.info('#' * 80)
                self.population.init_population()
        
            for epoch in range(generations):
                logger.info('*' * 80)
                logger.info(f'Starting epoch {epoch + 1}')
                self.step(epoch, generations, alpha=alpha)
                self.log_top_five()
                if self.population.bestindividual.fitness >= target:
                    logger.info(f'Target achieved: {self.population.bestindividual.fitness}')
                    logger.info('*' * 80)
                    break
            
            best_of_runs.append(self.population.bestindividual)
            logger.info(f"Run {run_idx + 1} Best Fitness: {self.population.bestindividual.fitness:.4f}")
            
        logger.info('*' * 80)
        
        if n_runs > 1:
            self.run_robustness_report(best_of_runs)
        
        self.log_tail()
        self.save_best_result()
        self.save_detailed_report()

    def save_detailed_report(self):
        if not (self.setup and hasattr(self.setup, 'project_folder') and self.population):
            return

        # 1. Save Description
        desc_path = os.path.join(self.setup.project_folder, "description.txt")
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write(self.setup.DESCRIPTION)
            
        # 2. Save Config Metadata
        if self.setup.METADATA:
            meta_path = os.path.join(self.setup.project_folder, "config_summary.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write("--- Experiment Configuration ---\n")
                for k, v in self.setup.METADATA.items():
                    if k != 'individual_config':
                        f.write(f"{k}: {v}\n")
                f.write(f"Final Seed used: {self.setup.RANDOM_SEED}\n")
                f.write(f"Penalty Factor: {self.setup.PENALTY_FACTOR}\n")
                f.write("-" * 30 + "\n")
                f.write("--- Individual Model Layout ---\n")
                f.write(json.dumps(self.setup.INDIVIDUAL_CONFIG, indent=4))
                f.write("\n" + "-" * 30 + "\n")
                
            json_meta_path = os.path.join(self.setup.project_folder, "config.evoconf")
            try:
                with open(json_meta_path, "w", encoding="utf-8") as f:
                    json.dump(self.setup.METADATA, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save .evoconf: {e}")
        
        # 3. Save Detailed Metrics CSV
        csv_path = os.path.join(self.setup.project_folder, "detailed_metrics.csv")
        fieldnames = ['rank', 'fitness', 'acc', 'f1', 'precision', 'recall', 'n_features', 'model_type', 'params']
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i, ind in enumerate(self.population.population):
                    # Extract model params if possible
                    model_name, params = "None", {}
                    if ind.model:
                        model_name = type(ind.model).__name__
                        try:
                            params = ind.model.get_params()
                        except Exception:
                            params = {"error": "Could not retrieve parameters"}
                    
                    writer.writerow({
                        'rank': i + 1,
                        'fitness': f"{ind.fitness:.6f}",
                        'acc': f"{ind.acc:.6f}" if ind.acc is not None else "0.0",
                        'f1': f"{ind.f1:.6f}" if ind.f1 is not None else "0.0",
                        'precision': f"{ind.prec:.6f}" if ind.prec is not None else "0.0",
                        'recall': f"{ind.recall:.6f}" if ind.recall is not None else "0.0",
                        'n_features': ind.radiomics.sum() if ind.radiomics is not None else 0,
                        'model_type': model_name,
                        'params': str(params)
                    })
            logger.info(f"Detailed metrics saved to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save detailed report: {e}")

    def run_robustness_report(self, best_individuals):
        logger.info("\n📊 FEATURE STABILITY REPORT 📊")
        logger.info("Checking which features consistently appear across different seeds...")
        
        all_features = np.array([ind.radiomics for ind in best_individuals])
        selection_counts = np.sum(all_features, axis=0)
        stability_ratio = selection_counts / len(best_individuals)
        
        stable_features = np.where(stability_ratio >= 0.7)[0]
        
        logger.info(f"Total runs: {len(best_individuals)}")
        logger.info(f"Features selected in >= 70% of runs: {len(stable_features)}")
        if len(stable_features) > 0:
            logger.info(f"Stable Feature Indices: {stable_features.tolist()}")
        
        # Log consistency of model selection
        models = [type(ind.model).__name__ for ind in best_individuals]
        unique_models, counts = np.unique(models, return_counts=True)
        logger.info("Model Selection Stability:")
        for m, c in zip(unique_models, counts):
            logger.info(f"  - {m}: {c}/{len(best_individuals)} runs")
        logger.info('*' * 80)

    def welcome(self):
        # Cross-platform clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        welcome_message = """
###############################################################################
#                                                                             #
#                      ░▒▓█ WELCOME TO EVO-FEATURESEL █▓▒░                    #
#                                                                             #
#                     🔥🔥🔥        MULTITHREAD       🔥🔥🔥               #
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
