import logging
import sys
import os
from typing import Iterable
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC

# Setup module-level logger
logger = logging.getLogger("evo.individual")

class BaseIndividual(ABC):
    def __init__(self,
                 filament_len: int,
                 genes: Iterable,
                 project_folder: str) -> None:
        super().__init__()

        # Check if DNA is binary
        if not all([x in (0, 1) for x in genes]):
            raise ValueError("DNA must contain only [0,1] integers")

        if not isinstance(filament_len, int):
            raise TypeError('DNA must be an integer')

        if len(genes) != filament_len:
            raise ValueError(
                f'Too many genes for the filament, or vice versa (filament_len == len(genes)), got {filament_len} and {len(genes)}')

        if not filament_len > 1:
            raise ValueError(
                f'Nothing to optimize got filament_len {filament_len}')
        
        self.filament_len: int = filament_len
        self.genes: np.ndarray = np.array(genes, dtype=np.int8)
        self.project_folder = project_folder

        # Set fitness to -1.0 as initial value
        self._fitness: float = -1.0

    @abstractmethod
    def fitness_eval(self, DATA: tuple, LABELS: tuple) -> float:
        pass

    def binaryToDecimal(self, binary):
        if len(binary) == 0:
            return 0
        return int("".join(map(str, map(int, binary))), 2)

    
    @property
    def fitness(self) -> float:
        return self._fitness

    def __str__(self) -> str:
        return f'{id(self)}:{self.fitness}'

    def __repr__(self):
        return f'{id(self)!r}:{self.fitness!r}'
    
    
class Individual(BaseIndividual):
    def __init__(self, filament_len, genes, bits: dict, project_folder, random_state) -> None:
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
        if not np.count_nonzero(self.genes):
            self._fitness = -1.0
            logger.warning(f'Every gene is zero... Killing individual {id(self)}')
            return

        try:
            X_train, X_test = DATA
            y_train, y_test = LABELS
                        
            self.radiomics, self.model_sel, self.model_param = self.to_phenotype()

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
                X_train_sel = X_train[:, np.array(self.radiomics, dtype=bool)]
                X_test_sel = X_test[:, np.array(self.radiomics, dtype=bool)]
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
        genes = self.genes[:self.bits['features']]
        model_selection = self.binaryToDecimal(
            self.genes[self.bits['features']:self.bits['features'] + self.bits['model_selection']]
        )
        param_bits = self.genes[self.bits['features'] + self.bits['model_selection']:]
        
        model_param = dict()

        match (model_selection):
            case 0:
                n_estimators = self.binaryToDecimal(param_bits[:9])
                model_param['n_estimators'] = n_estimators if n_estimators > 2 else 2
                criterion_selector = self.binaryToDecimal(param_bits[9:])
                model_param['criterion'] = 'gini' if criterion_selector == 0 else ('entropy' if criterion_selector == 1 else 'log_loss')
            case 1:
                parteintera = 1
                mantissa = self.binaryToDecimal(param_bits[:3]) * (10**-1)
                segno = 1.0 if self.binaryToDecimal([param_bits[3]]) == 0 else -1.0
                esponente = self.binaryToDecimal(param_bits[4:7])
                model_param['C'] = (parteintera + mantissa) * (10 ** (segno * esponente))
                
                kernel_selector = self.binaryToDecimal(param_bits[8:10])
                kernels = ['linear', 'poly', 'rbf', 'sigmoid']
                model_param['kernel'] = kernels[kernel_selector] if kernel_selector < 4 else 'rbf'
                model_param['degree'] = self.binaryToDecimal(param_bits[10:]) + 1
            case 2:
                n_estimators = self.binaryToDecimal(param_bits[:9])
                model_param['n_estimators'] = n_estimators if n_estimators > 2 else 2
                model_param['criterion'] = 'friedman_mse' if self.binaryToDecimal(param_bits[9:10]) == 0 else 'squared_error'
                model_param['loss'] = 'log_loss' if self.binaryToDecimal(param_bits[10:]) == 0 else 'exponential'
            case 3:
                n_estimators = self.binaryToDecimal(param_bits[:9])
                model_param['n_estimators'] = n_estimators if n_estimators > 2 else 2
                criterion_selector = self.binaryToDecimal(param_bits[9:])
                model_param['criterion'] = 'gini' if criterion_selector == 0 else ('entropy' if criterion_selector == 1 else 'log_loss')
        
        return genes, model_selection, model_param


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
