from typing import Iterable
from abc import ABC, abstractmethod
import numpy as np
from evo.utils import Setup


class BaseIndividual(ABC):
    def __init__(self,
                 filament_len: int,
                 genes: Iterable) -> None:
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
        
        self.filament_len: int = filament_len
        self.genes: Iterable = np.array(genes, dtype=np.int8)

        # Set fitness to -1.0 as initial value
        self._fitness: float = -1.0

    @abstractmethod
    def fitness_eval(self,) -> float:
        pass

    @property
    def fitness(self,) -> float:
        return self._fitness

    def __str__(self) -> str:
        return f'DNA: {self.genes}'


class Individual(BaseIndividual):
    def __init__(self, filament_len, genes) -> None:
        super().__init__(filament_len, genes)

    def fitness_eval(self, FEATURES: tuple, LABELS: tuple, model) -> float:
        '''
        Function that evaluates the individual fitness.

        Parameters:
            FEATURES (`tuple(X_train, X_test)`): sets that will be sliced from the individual genes
            LABELS   (`tuple(y_train, y_test)`): labels to fit the model and evaluate fitness
            model    (`BaseEstimator`): model for fitness evaluation
        Returns:
            `float` : The fitness value        
        '''

        if not np.count_nonzero(self.genes):
            # @TODO: Should be like killing the individual e.g. place fitness = -1
            self._fitness = -1.0
            raise Warning(f'All genes are zero... Killing individual {id(self)}')

        try:
            # @TODO: Implement model selection and model parameter(s) selection
            X_train,X_test = FEATURES
            y_train,y_test = LABELS
            
            radiomics   = self.genes[:107]      # array[107]
            model_sel   = self.genes[107:107+4] # array[2]
            model_param = self.genes[107+4:]    # array[6]
                        
            model_param = self.genes[107+4:]
            
            if self.binaryToDecimal(model_sel) == 0:
                pass
            
            
            X_train_sel = X_train[:, np.array(radiomics, dtype=bool)]
            X_test_sel  = X_test [:, np.array(radiomics, dtype=bool)]
            
            result = model.fit(X_train_sel,y_train)
        except:
            self._fitness = -1.0
            raise Warning(f'All genes are zero... Killing individual {id(self)}')

        preds = model.predict(X_test_sel)
        self._fitness = mcc_score(y_test,preds)
                    
        def binaryToDecimal(self,binary):
            decimal, i = 0, 0
            while(binary != 0):
                dec = binary % 10
                decimal = decimal + dec * pow(2, i)
                binary = binary//10
                i += 1
            return decimal
 
        
        
        
        


if __name__ == '__main__':
    genes = np.array((1, 0, 0, 1, 1, 0, 1, 0))
    filament_len = len(genes)

    evo_setup = Setup()
    # should be: evo_setup.FEATURES = (X_train, X_test) ready for sk-learn fit()
    evo_setup.FEATURES = np.random.randint(0, 2, (50, filament_len))

    individual = Individual(filament_len=filament_len,
                            genes=genes)

    print(individual.fitness)

    individual.fitness_eval(evo_setup.FEATURES)

    print(individual.fitness)

    # Check if DNA is binary @TODO: evo for floating points
    print(Individual(filament_len=8,
                     genes=(1, 0, 0, 1, 1, 0, 1, 0)
                     )
          )
