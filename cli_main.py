import os
import argparse
import logging
import numpy as np

from evo.utils import Setup, preprocessing_general, get_default_individual_config
from evo.population import Population
from evo.runner import Runner

def main():
    parser = argparse.ArgumentParser(description="evo-featuresel CLI")
    
    # Dataset arguments
    parser.add_argument('--train_feat', type=str, default='mock_train_features.csv', help='Path to train features CSV')
    parser.add_argument('--train_labels', type=str, default='mock_train_labels.csv', help='Path to train labels CSV')
    parser.add_argument('--val_feat', type=str, default='mock_valid_features.csv', help='Path to validation features CSV')
    parser.add_argument('--val_labels', type=str, default='mock_valid_labels.csv', help='Path to validation labels CSV')
    parser.add_argument('--test_feat', type=str, default='', help='Path to test features CSV')
    parser.add_argument('--test_labels', type=str, default='', help='Path to test labels CSV')
    
    # Algorithm arguments
    parser.add_argument('--pop_size', type=int, default=50, help='Population size')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--alpha', type=float, default=0.5, help='Mutation alpha')
    parser.add_argument('--penalty', type=float, default=0.01, help='Feature selection penalty factor')
    parser.add_argument('--patience', type=int, default=300, help='Max seconds per individual evaluation')
    
    # Preprocessing arguments
    parser.add_argument('--pca', action='store_true', help='Apply PCA (95%% variance)')
    parser.add_argument('--lda', action='store_true', help='Apply LDA')
    parser.add_argument('--scaler', type=str, default='Standard', choices=['Standard', 'MinMax'], help='Scaler type')
    
    # Output arguments
    parser.add_argument('--exp_folder', type=str, default='experiment', help='Base experiment folder')
    parser.add_argument('--prefix', type=str, default='cli_run_', help='Project folder prefix')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    if not os.path.exists(args.train_feat):
        logging.error(f"Train features file not found: {args.train_feat}")
        return

    # 1. Data Loading and Preprocessing
    logging.info("📥 Loading and preprocessing data...")
    try:
        data_all, labels_all = preprocessing_general(
            args.train_feat, 
            args.val_feat if args.val_feat else None, 
            args.test_feat if args.test_feat else None,
            args.train_labels if args.train_labels else None,
            args.val_labels if args.val_labels else None,
            args.test_labels if args.test_labels else None,
            pca=args.pca,
            lda=args.lda,
            scaler_type=args.scaler
        )
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return

    X_train, X_val, X_test = data_all
    y_train, y_val, y_test = labels_all
    
    # Decide which split to use for fitness evaluation
    if X_val is None:
        if X_test is not None:
            data_evo = (X_train, X_test)
            labels_evo = (y_train, y_test)
        else:
            data_evo = (X_train, X_train)
            labels_evo = (y_train, y_train)
    else:
        data_evo = (X_train, X_val)
        labels_evo = (y_train, y_val)

    # 2. Configure Setup
    setup = Setup(
        project_prefix=args.prefix,
        experiment_folder=args.exp_folder,
        use_timestamp=True,
        DESCRIPTION="CLI Evolutionary Feature Selection",
        INDIVIDUAL_CONFIG=get_default_individual_config()
    )
    
    setup.POP_SIZE = args.pop_size
    setup.PENALTY_FACTOR = args.penalty
    setup.PATIENCE = args.patience
    setup.DATA = data_evo
    setup.LABELS = labels_evo
    setup.RANDOM_SEED = args.seed
    
    # Initialize bits array and compute filament length
    setup.BITS = {'features': X_train.shape[1]}
    setup.calculate_filament_len()
    
    setup.seed_all(setup.RANDOM_SEED)
    setup.init_rng()

    # 3. Evolution Loop
    pop = Population(setup=setup)
    runner = Runner(setup=setup, population=pop)
    
    logging.info("🐣 Initializing Population 🐤")
    pop.init_population()

    logging.info(f"🚀 Starting Evolution: {args.generations} Generations, Pop Size {args.pop_size}")
    for epoch in range(args.generations):
        runner.step(epoch, args.generations, alpha=args.alpha)
        runner.log_top_five()

    # 4. Finalizing
    runner.log_tail()
    runner.save_best_result()
    runner.save_detailed_report()
    
    best = pop.best_individual
    best.ensure_phenotype()
    logging.info("✅ EVOLUTION COMPLETED")
    logging.info(f"🏆 Best MCC: {best.fitness:.4f}")
    logging.info(f"🎯 Best Model: {type(best.model).__name__}")
    logging.info(f"🧬 Selected Features: {int(best.radiomics.sum())} / {setup.BITS['features']}")
    logging.info(f"📁 Results saved in: {setup.project_folder}")

if __name__ == '__main__':
    main()
