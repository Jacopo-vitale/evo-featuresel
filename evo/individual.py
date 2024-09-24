from typing import Iterable
from abc import ABC, abstractmethod
import numpy as np
from evo.utils import Setup
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import logging
import sys,os

class BaseIndividual(ABC):
    def __init__(self,
                 filament_len: int,
                 genes: Iterable,
                 project_folder : str) -> None:
        super().__init__()

        # Check if DNA is binary @TODO: evo for floating points
        if not all([x in (0, 1) for x in genes]):
            raise Exception("DNA must contain only [0,1] integers")

        if type(filament_len) is not int:
            raise TypeError('DNA must be an integer')

        if len(genes) != filament_len:
            raise Exception(
                f'Too many genes for the filament, or vice versa (filament_len == len(genes)), got {filament_len} and {len(genes)}')

        if not filament_len > 1:
            raise Exception(
                f'Nothing to optimize got filament_len {filament_len}')
        self.logger = logging.getLogger("individual")
        self.logger.setLevel(logging.DEBUG)
        # Create handlers for logging to the standard output and a file
        stdoutHandler = logging.StreamHandler(stream=sys.stdout)
        errHandler = logging.FileHandler(os.path.join(project_folder,"error.log"))
        # Set the log levels on the handlers
        stdoutHandler.setLevel(logging.DEBUG)
        errHandler.setLevel(logging.ERROR)
        # Create a log format using Log Record attributes
        fmt = logging.Formatter(
            "%(message)s"
        )

        # Set the log format on each handler
        stdoutHandler.setFormatter(fmt)
        errHandler.setFormatter(fmt)

        # Add each handler to the Logger object
        self.logger.addHandler(stdoutHandler)
        self.logger.addHandler(errHandler)
        
        self.filament_len: int = filament_len
        self.genes: Iterable = np.array(genes, dtype=np.int8)

        # Set fitness to -1.0 as initial value
        self._fitness: float = -1.0

    @abstractmethod
    def fitness_eval(self,) -> float:
        pass

    def binaryToDecimal(self,binary):
        decimal : np.int16 = 0
        for p,b in enumerate(binary,start=1):
            decimal += b * (2**(len(binary) - p))
        return decimal
    
    @property
    def fitness(self,) -> float:
        return self._fitness

    def __str__(self) -> str:
        return f'DNA: {self.genes}'


class Individual(BaseIndividual):
    def __init__(self, filament_len, genes, bits : dict, project_folder) -> None:
        super().__init__(filament_len, genes,project_folder)
        
        self.bits = bits
        

    def fitness_eval(self, DATA: tuple, LABELS: tuple,) -> float:
        '''
        Function that evaluates the individual fitness.

        Parameters:
            DATA     (`tuple(X_train, X_test)`): sets that will be sliced from the individual genes
            LABELS   (`tuple(y_train, y_test)`): labels to fit the model and evaluate fitness
            model    (`BaseEstimator`): model for fitness evaluation
        Returns:
            `float` : The fitness value        
        '''

        if not np.count_nonzero(self.genes):
            # killing the individual e.g. place fitness = -1
            self._fitness = -1.0
            raise self.logger.warning(f'All genes are zero... Killing individual {id(self)}')

        try:
            X_train,X_test = DATA
            y_train,y_test = LABELS
                        
            radiomics, model_sel, model_param  = self.to_phenotype()


            match (model_sel):
                case 0:
                    model = RandomForestClassifier()
                case 1:
                    model = RandomForestClassifier()
                case 2:
                    model = RandomForestClassifier()
                case 3:
                    model = RandomForestClassifier()
                
            
            X_train_sel = X_train[:, np.array(radiomics, dtype=bool)]
            X_test_sel  = X_test [:, np.array(radiomics, dtype=bool)]
            
            model.fit(X_train_sel,y_train)
            preds = model.predict(X_test_sel)
            self._fitness = matthews_corrcoef(y_test,preds)
        
        except:
            self._fitness = -1.0
            self.logger.warning(f'All genes are zero... Killing individual {id(self)}')

    def to_phenotype(self,):
        return (    
                    self.genes[:self.bits['features']], # radiomics to be selected

                    self.binaryToDecimal(
                        self.genes[self.bits['features']:\
                                   self.bits['features'] + self.bits['model_selection']
                            ]), # Selected Model
                    self.binaryToDecimal(
                        self.genes[self.bits['features'] + self.bits['model_selection']:]
                        ), # Model parameter(s)
                )
 

if __name__ == '__main__':
    genes = np.random.choice([0,1],size=150)
    filament_len = len(genes)

    evo_setup = Setup()
    # should be: evo_setup.FEATURES = (X_train, X_test) ready for sk-learn fit()
    evo_setup.DATA = np.random.randint(0, 2, (50, filament_len))

    individual = Individual(filament_len=filament_len,
                            genes=genes)

    print(individual.fitness)

    individual.fitness_eval(evo_setup.DATA)

    print(individual.fitness)

    # Check if DNA is binary @TODO: evo for floating points
    print(Individual(filament_len=8,
                     genes=(1, 0, 0, 1, 1, 0, 1, 0)
                     )
          )
