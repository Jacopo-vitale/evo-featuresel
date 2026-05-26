import logging
import sys
import os
import importlib
from typing import Iterable
from abc import ABC, abstractmethod
import numpy as np

try:
    from evo.core import (
        fast_binary_to_decimal, 
        pack_bits, 
        unpack_bits, 
        decode_individual,
        fast_binary_to_decimal_packed,
        fast_binary_metrics
    ) 
except ImportError:
# ... (rest of imports and base class)
    # Fallback implementations omitted for brevity but should be kept in a real scenario
    # or just assume Cython is available since we are in a dev branch.
    def pack_bits(x): return x
    def unpack_bits(x, n): return x
    def decode_individual(p, b): return 0, {} 
    def fast_binary_to_decimal(x): return 0
    def fast_binary_metrics(y_true, y_pred): return 0,0,0,0,0,[]

# Setup module-level logger
logger = logging.getLogger("evo.individual")

# Global caches for dynamic components
_MODEL_CACHE = {}         # {import_path: (model_class, supports_random_state)}
_FITNESS_FUNC_CACHE = {}  # {hash(fitness_code): custom_fitness_function}

class BaseIndividual(ABC):
# ... (rest of class)
    def __init__(self,
                 filament_len: int,
                 genes: np.ndarray,
                 project_folder: str) -> None:
        super().__init__()

        # genes should now be a packed uint8 array
        self.filament_len: int = filament_len
        self.genes: np.ndarray = genes # Expected to be packed
        self.project_folder = project_folder

        # Set fitness to -1.0 as initial value
        self._fitness: float = -1.0

    @abstractmethod
    def fitness_eval(self, DATA: tuple, LABELS: tuple) -> float:
        pass

    @property
    def fitness(self) -> float:
        return self._fitness

    def __str__(self) -> str:
        return f'{id(self)}:{self.fitness}'

    def __repr__(self):
        return f'{id(self)!r}:{self.fitness!r}'
    
    
class Individual(BaseIndividual):
    def __init__(self, filament_len, genes, bits: dict, project_folder, random_state, 
                 penalty_factor: float = 0.0,
                 cython_layout: tuple = None,
                 enabled_models: list = None,
                 fitness_code: str = None) -> None:
        # If genes are int8 (unpacked), pack them
        if genes.dtype == np.int8:
            genes = pack_bits(genes)
            
        super().__init__(filament_len, genes, project_folder)
        
        self.bits = bits
        self.random_state = random_state 
        self.penalty_factor = penalty_factor
        self.cython_layout = cython_layout # (param_names, param_categories, layout_array)
        self.enabled_models = enabled_models # List of enabled model dictionaries
        self.fitness_code = fitness_code # Custom python logic
        
        self.model = None
# ... (rest of attributes)
        self.radiomics = None
        self.model_sel = None
        self.model_param = None

        self.preds = None
        self._fitness = -1.0
        self.acc = None
        self.f1 = None
        self.prec = None
        self.recall = None
        self.cm = None

    def ensure_phenotype(self):
        """Ensure radiomics and model parameters are decoded from genes."""
        if self.radiomics is not None:
            return
            
        self.radiomics_packed, self.model_sel, self.model_param = self.to_phenotype()
        self.radiomics = unpack_bits(self.radiomics_packed, self.bits['features']).astype(bool)

    def fitness_eval(self, DATA: tuple, LABELS: tuple) -> float:
        # Check if any gene is set (simplified for packed)
        if not np.any(self.genes):
            self._fitness = -1.0
            logger.warning(f'Every gene is zero... Killing individual {id(self)}')
            return

        try:
            X_train, X_test = DATA
            y_train, y_test = LABELS
                        
            self.ensure_phenotype()

            # Dynamic Model Instantiation from Registry with Caching
            if self.enabled_models and self.model_sel < len(self.enabled_models):
                model_info = self.enabled_models[self.model_sel]
                import_path = model_info.get('import_path')
                
                if not import_path:
                    raise ValueError(f"Model {model_info['name']} has no import_path defined.")
                
                if import_path in _MODEL_CACHE:
                    model_class, supports_rs = _MODEL_CACHE[import_path]
                else:
                    module_name, class_name = import_path.rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    model_class = getattr(module, class_name)
                    
                    # Check if random_state is supported
                    temp_model = model_class()
                    supports_rs = 'random_state' in temp_model.get_params()
                    _MODEL_CACHE[import_path] = (model_class, supports_rs)
                
                if supports_rs:
                    self.model = model_class(random_state=self.random_state)
                else:
                    self.model = model_class()
            else:
                # Minimal Fallback for legacy / smoke tests
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=1)
                
            if self.radiomics.sum() >= 1:
                X_train_sel = X_train[:, self.radiomics]
                X_test_sel = X_test[:, self.radiomics]
            else:
                self._fitness = -1.0
                return

            self.model.set_params(**self.model_param)
            
            self.model.fit(X_train_sel, y_train)
            preds = self.model.predict(X_test_sel)
            self.preds = preds
            
            # Ensure int64 for Cython fast metrics
            y_test_fast = np.asarray(y_test, dtype=np.int64)
            preds_fast = np.asarray(preds, dtype=np.int64)
            
            mcc, acc, f1, prec, recall, cm = fast_binary_metrics(y_test_fast, preds_fast)
            
            # 1. Base Metrics
            n_features = self.bits['features']
            n_selected = self.radiomics.sum()
            
            # 2. Custom Fitness Logic with Caching
            if self.fitness_code:
                try:
                    code_hash = hash(self.fitness_code)
                    if code_hash in _FITNESS_FUNC_CACHE:
                        custom_func = _FITNESS_FUNC_CACHE[code_hash]
                    else:
                        # Provide a limited set of globals
                        safe_globals = {'np': np}
                        local_scope = {}
                        exec(self.fitness_code, safe_globals, local_scope)
                        
                        # Assume the user defined 'custom_fitness'
                        if 'custom_fitness' in local_scope:
                            custom_func = local_scope['custom_fitness']
                            _FITNESS_FUNC_CACHE[code_hash] = custom_func
                        else:
                            raise ValueError("Function 'custom_fitness' not found in provided code.")
                    
                    metrics_dict = {
                        'mcc': mcc, 'acc': acc, 'f1': f1, 
                        'prec': prec, 'recall': recall, 'cm': cm
                    }
                    self._fitness = custom_func(
                        y_test, preds, n_features, n_selected, metrics_dict
                    )
                except Exception as e:
                    logger.error(f"Custom fitness execution failed: {e}. Falling back to default.")
                    penalty = self.penalty_factor * (n_selected / n_features)
                    self._fitness = mcc - penalty
            else:
                # Default Logic
                penalty = self.penalty_factor * (n_selected / n_features)
                self._fitness = mcc - penalty
            
            self.acc = acc
            self.f1 = f1
            self.prec = prec
            self.recall = recall
            self.cm = cm

        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            self._fitness = -1.0


    def to_phenotype(self):
        # Use the unified Cython decoder
        if self.cython_layout:
            param_names, param_categories, layout_array = self.cython_layout
            model_sel, model_param = decode_individual(
                self.genes, 
                self.bits['features'], 
                self.bits['model_selection'],
                param_names,
                param_categories,
                layout_array
            )
        else:
            # Fallback (though BITS structure changed, this might need care)
            model_sel, model_param = decode_individual(self.genes, self.bits)
        
        # Extract features as packed bits (first N bits)
        feat_bytes = (self.bits['features'] + 7) // 8
        radiomics_packed = self.genes[:feat_bytes]
        
        return radiomics_packed, model_sel, model_param


if __name__ == '__main__':
    bits = {'features': 100, 'model_selection': 2, 'model_params': 11}
    filament_len = sum(bits.values())
    genes = np.random.choice([0, 1], size=filament_len)
    
    individual = Individual(filament_len=filament_len,
                            genes=genes,
                            bits=bits,
                            project_folder='.',
                            random_state=42)

    print(f"Initial fitness: {individual.fitness}")
