import numpy as np
import os
import sys
import random
import json
import datetime as dt
import pandas as pd
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from evo.models_registry import MODELS_REGISTRY

def get_max_workers():
    """
    Returns a safe number of workers for ProcessPoolExecutor, 
    capping at 61 on Windows to avoid OS limitations.
    """
    num_cpus = os.cpu_count() or 1
    if sys.platform == "win32":
        return min(num_cpus, 61)
    return num_cpus

def preprocessing_general(train_path, val_path, test_path,
# ... (rest of the function)
                          train_labels_path=None, val_labels_path=None, test_labels_path=None,
                          pca=False, lda=False, scaler_type="Standard"):
    """
    Generalized preprocessing that loads separate files for train, val, and test.
    If label paths are not provided, it assumes the last column is labels.
    """
    def load_and_split(feat_path, label_path=None):
        if not feat_path or not os.path.exists(feat_path):
            return None, None

        df_feat = pd.read_csv(feat_path)

        if label_path and os.path.exists(label_path):
            X = df_feat.to_numpy()
            y = pd.read_csv(label_path).iloc[:, 0].to_numpy() # Assumes single-column label file
        else:
            # Assume last column is labels
            X = df_feat.iloc[:, :-1].to_numpy()
            y = df_feat.iloc[:, -1].to_numpy()

        return X, y

    X_train, y_train = load_and_split(train_path, train_labels_path)
    X_val, y_val = load_and_split(val_path, val_labels_path)
    X_test, y_test = load_and_split(test_path, test_labels_path)

    if X_train is None:
        raise ValueError("Train dataset is required.")

    imputer = SimpleImputer()
    if scaler_type == "MinMax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))

    if X_val is not None:
        X_val = scaler.transform(imputer.transform(X_val))
    if X_test is not None:
        X_test = scaler.transform(imputer.transform(X_test))

    # Apply PCA or LDA if requested
    if pca:
        pca_model = PCA(n_components=0.95) # Keep 95% of variance
        X_train = pca_model.fit_transform(X_train)
        if X_val is not None:
            X_val = pca_model.transform(X_val)
        if X_test is not None:
            X_test = pca_model.transform(X_test)
    elif lda:
        lda_model = LDA()
        X_train = lda_model.fit_transform(X_train, y_train)
        if X_val is not None:
            X_val = lda_model.transform(X_val)
        if X_test is not None:
            X_test = lda_model.transform(X_test)

    return ((X_train, X_val, X_test), (y_train, y_val, y_test))

def get_default_individual_config():
    """
    Returns the default configuration for models and their parameters.
    """
    return {
        "models": copy.deepcopy(MODELS_REGISTRY)
    }

@dataclass
class Setup:
    """
    Setup class for the evolutionary algorithm experiment.
    """
    POP_SIZE: int = 50
    MUT_RATE: float = 1.0
    GENES: List[int] = field(default_factory=lambda: [0, 1])
    FILAMENT_LEN: int = 0
    DATA: Optional[Tuple[np.ndarray, np.ndarray]] = None
    LABELS: Optional[Tuple[np.ndarray, np.ndarray]] = None
    BITS: Dict[str, int] = field(default_factory=dict)
    DESCRIPTION: str = "Evolutionary Feature Selection Experiment"
    RANDOM_SEED: int = 42
    PENALTY_FACTOR: float = 0.01  # Penalty for each selected feature as a ratio
    PATIENCE: int = 300  # Maximum seconds an individual is allowed to evaluate before timeout
    N_ROBUSTNESS_RUNS: int = 1    # Number of runs for stability check
    METADATA: Dict[str, Any] = field(default_factory=dict) # Store arbitrary config metadata
    
    # New: Individual configuration
    INDIVIDUAL_CONFIG: Dict[str, Any] = field(default_factory=get_default_individual_config)
    FITNESS_CODE: Optional[str] = None
    
    experiment_folder: str = 'experiment'
    project_prefix: str = ''
    use_timestamp: bool = True
    
    def __post_init__(self):
        folder_name = self.project_prefix
        if self.use_timestamp:
            folder_name += dt.datetime.now().strftime('%Y%m%d%H%M')
        
        if not folder_name:
            folder_name = "latest_run"

        self.project_folder = os.path.join(
            self.experiment_folder,
            folder_name
        )
        os.makedirs(self.experiment_folder, exist_ok=True)
        os.makedirs(self.project_folder, exist_ok=True)
        self.seed_all(self.RANDOM_SEED)

    def calculate_filament_len(self):
        """
        Calculates the filament length and BITS dictionary based on INDIVIDUAL_CONFIG.
        """
        enabled_models = [m for m in self.INDIVIDUAL_CONFIG['models'] if m['enabled']]
        if not enabled_models:
            raise ValueError("At least one model must be enabled in the configuration.")

        # 1. Features bits (must be set externally as it depends on dataset)
        n_features = self.BITS.get('features', 0)
        
        # 2. Model selection bits
        n_models = len(enabled_models)
        model_sel_bits = int(np.ceil(np.log2(n_models))) if n_models > 1 else 0
        
        # 3. Model parameters bits
        # The filament must be large enough to hold the parameters of the model with most bits
        max_param_bits = 0
        for model in enabled_models:
            model_bits = 0
            for p_name, p_data in model['params'].items():
                if p_data['enabled']:
                    if p_data['type'] == 'float':
                        model_bits += p_data['bits_mantissa'] + p_data['bits_exponent'] + p_data['bits_sign']
                    else:
                        model_bits += p_data['bits']
            if model_bits > max_param_bits:
                max_param_bits = model_bits
        
        self.BITS['model_selection'] = model_sel_bits
        self.BITS['model_params'] = max_param_bits
        self.FILAMENT_LEN = n_features + model_sel_bits + max_param_bits
        return self.FILAMENT_LEN

    def get_cython_layout(self):
        """
        Prepares the flat layout arrays for the Cython decoder.
        Returns: (param_names, param_categories, model_layouts_array)
        """
        enabled_models = [m for m in self.INDIVIDUAL_CONFIG['models'] if m['enabled']]
        
        # Collect all unique parameter names and their possible categorical values
        all_param_names = []
        all_categories = []
        
        for model in enabled_models:
            for p_name, p_data in model['params'].items():
                if p_data['enabled']:
                    if p_name not in all_param_names:
                        all_param_names.append(p_name)
                    if p_data['type'] == 'categorical':
                        if p_data['values'] not in all_categories:
                            all_categories.append(p_data['values'])

        # Build the model_layouts array (2D: num_enabled_models x MAX_PARAMS_INFO)
        # Max params per model info structure: [num_params, p1_name_idx, p1_bits, p1_type, p1_e1, p1_e2, p1_e3, ...]
        max_params = max(len([p for p in m['params'].values() if p['enabled']]) for m in enabled_models)
        layout_width = 1 + max_params * 6
        layout_array = np.zeros((len(enabled_models), layout_width), dtype=np.int32)

        for i, model in enumerate(enabled_models):
            enabled_params = [(name, data) for name, data in model['params'].items() if data['enabled']]
            layout_array[i, 0] = len(enabled_params)
            for j, (p_name, p_data) in enumerate(enabled_params):
                col = 1 + j * 6
                layout_array[i, col] = all_param_names.index(p_name)
                
                # Types: 0=int, 1=categorical, 2=float
                if p_data['type'] == 'int':
                    layout_array[i, col+1] = p_data['bits']
                    layout_array[i, col+2] = 0
                    layout_array[i, col+3] = p_data['min']
                elif p_data['type'] == 'categorical':
                    layout_array[i, col+1] = p_data['bits']
                    layout_array[i, col+2] = 1
                    layout_array[i, col+3] = all_categories.index(p_data['values'])
                elif p_data['type'] == 'float':
                    total_bits = p_data['bits_mantissa'] + p_data['bits_exponent'] + p_data['bits_sign']
                    layout_array[i, col+1] = total_bits
                    layout_array[i, col+2] = 2
                    layout_array[i, col+3] = p_data['bits_mantissa']
                    layout_array[i, col+4] = p_data['bits_exponent']
                    layout_array[i, col+5] = p_data['bits_sign']

        return all_param_names, all_categories, layout_array

    def seed_all(self, seed: int):
        """
        Seeds all random number generators for reproducibility.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)

    def init_rng(self):
        # Kept for backward compatibility
        self.seed_all(self.RANDOM_SEED)

if __name__ == '__main__':
    evo_setup = Setup(POP_SIZE=500)
    print(f"Project folder: {evo_setup.project_folder}")
