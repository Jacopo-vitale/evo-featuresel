import logging
import sys
import os
from typing import Iterable
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC

try:
    from evo.core import (
        fast_binary_to_decimal, 
        pack_bits, 
        unpack_bits, 
        decode_individual,
        fast_binary_to_decimal_packed
    )
except ImportError:
    # Fallback implementations omitted for brevity but should be kept in a real scenario
    # or just assume Cython is available since we are in a dev branch.
    def pack_bits(x): return x
    def unpack_bits(x, n): return x
    def decode_individual(p, b): return 0, {} 
    def fast_binary_to_decimal(x): return 0

# Setup module-level logger
logger = logging.getLogger("evo.individual")

class BaseIndividual(ABC):
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
    def __init__(self, filament_len, genes, bits: dict, project_folder, random_state) -> None:
        # If genes are int8 (unpacked), pack them
        if genes.dtype == np.int8:
            genes = pack_bits(genes)
            
        super().__init__(filament_len, genes, project_folder)
        
        self.bits = bits
        self.random_state = random_state 

        self.model = None
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

    def fitness_eval(self, DATA: tuple, LABELS: tuple) -> float:
        # Check if any gene is set (simplified for packed)
        if not np.any(self.genes):
            self._fitness = -1.0
            logger.warning(f'Every gene is zero... Killing individual {id(self)}')
            return

        try:
            X_train, X_test = DATA
            y_train, y_test = LABELS
                        
            self.radiomics_packed, self.model_sel, self.model_param = self.to_phenotype()
            
            # For radiomics, we need the unpacked boolean mask for sklearn
            self.radiomics = unpack_bits(self.radiomics_packed, self.bits['features']).astype(bool)

            match (self.model_sel):
                case 0:
                    self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=1)
                case 1:
                    self.model = SVC(random_state=self.random_state)
                case 2:
                    self.model = GradientBoostingClassifier(random_state=self.random_state)
                case 3:
                    self.model = ExtraTreesClassifier(random_state=self.random_state, n_jobs=1)
                case _:
                    raise ValueError(f"Unknown model selection: {self.model_sel}")
                
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
            self._fitness = matthews_corrcoef(y_test, preds)
            self.acc = accuracy_score(y_test, preds)
            self.f1 = f1_score(y_test, preds)
            self.prec = precision_score(y_test, preds, zero_division=0.0)
            self.recall = recall_score(y_test, preds)
            self.cm = confusion_matrix(y_test, preds)

        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            self._fitness = -1.0


    def to_phenotype(self):
        # Use the unified Cython decoder
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
