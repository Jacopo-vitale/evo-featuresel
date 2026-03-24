import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import load

from evo.utils import Setup
from evo.population import Population
from evo.runner import Runner

def preprocessing(train_subj, test_subj, dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    # Assuming 'subj' is the column for subject IDs
    # and labels are in the second to last column, and some other info in the last
    # This might need adjustment based on the actual CSV structure
    X_train = df.query(f'subj in {train_subj}').iloc[:, :-2].to_numpy()
    y_train = df.query(f'subj in {train_subj}').iloc[:, -2].to_numpy()
    X_test = df.query(f'subj in {test_subj}').iloc[:, :-2].to_numpy()
    y_test = df.query(f'subj in {test_subj}').iloc[:, -2].to_numpy()

    imputer, scaler = SimpleImputer(), StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))
    return ((X_train, X_test), (y_train, y_test))

def main():
    # Default parameters
    train_subj = [2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17]
    test_subj = [5, 9, 10]
    
    # Try to use a relative path first
    dataset_path = 'data/dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}. Please provide a valid path.")
        # Create a dummy dataset if it doesn't exist for demonstration purposes
        os.makedirs('data', exist_ok=True)
        cols = [f'feat_{i}' for i in range(20)] + ['subj', 'label', 'extra']
        dummy_data = np.random.randn(100, 20)
        subjs = np.random.choice(train_subj + test_subj, 100)
        labels = np.random.choice([0, 1], 100)
        extra = np.random.randn(100)
        df_dummy = pd.DataFrame(dummy_data, columns=cols[:-3])
        df_dummy['subj'] = subjs
        df_dummy['label'] = labels
        df_dummy['extra'] = extra
        df_dummy.to_csv(dataset_path, index=False)
        print(f"Created a dummy dataset at {dataset_path} for demonstration.")

    # Data loading and preproc
    try:
        data, labels = preprocessing(train_subj, test_subj, dataset_path)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    setup = Setup(project_prefix='experiment_')
    
    setup.POP_SIZE = 50
    setup.BITS = {
        'features': data[0].shape[1],
        'model_selection': 2,
        'model_params': 11,
    }
    
    setup.FILAMENT_LEN = sum(setup.BITS.values())
    setup.GENES = [0, 1]
    setup.DATA = data
    setup.LABELS = labels
    setup.RANDOM_SEED = 42
    setup.DESCRIPTION = f'Evolutionary feature selection on subjects {test_subj}'
    setup.init_rng()
    
    pop = Population(setup=setup)
    r = Runner(setup=setup, population=pop)
    r.run(generations=10) # Reduced for quick test

if __name__ == '__main__':
    main()


    