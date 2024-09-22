from typing import Iterable,overload,Any
from abc import ABC,abstractmethod
import numpy as np
from utils import Setup

class BaseIndividual(ABC):
    def __init__(self, 
                 filament_len : int, 
                 genes : Iterable) -> None:
        super().__init__()
        
        # Check if DNA is binary @TODO: evo for floating points
        if not all([x in (0,1) for x in genes]):
            raise Exception("DNA must contain only [0,1] integers")
        
        if type(filament_len) is not int:
            raise TypeError('DNA must be an integer')

        if len(genes) != filament_len:
            raise Exception(f'Too many genes for the filament, or vice versa (filament_len == len(genes)), got {filament_len} and {len(genes)}')

        if not filament_len > 1:
            raise Exception(f'Nothing to optimize got filament_len {filament_len}')

        self.filament_len : int = filament_len
        self.genes : Iterable = np.array(genes,dtype=np.int8)

        # Set fitness to -1.0 as initial value
        self._fitness : float = -1.0

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
    

    def fitness_eval(self, FEATURES, LABELS=None, model = None) -> float:
        '''
        Function that evaluates the individual fitness.

        Parameters:
            FEATURES (`tuple(X_train, X_test)`): sets that will be sliced from the individual genes
            LABELS   (`tuple(y_train, y_test)`): labels to fit the model and evaluate fitness
          
        Returns:
            `float` : The fitness value        
        '''

        if not np.count_nonzero(self.genes):
            print(self.genes)
            raise Exception('All genes are zero... ABORTING')
        
        try:
            #@TODO: Implement model selection and model parameter(s) selection
            selected_features = FEATURES[:,np.array(self.genes,dtype=bool)]
            #result = model.fit(selected_features,LABELS)
        except:
            raise Exception('Model can\'t be fitted... ABORTING')
        
        # self._fitness = metric(TEST_SET, TEST_LABELS)
        self._fitness = selected_features.sum()
        
        return self._fitness


if __name__ == '__main__':
    genes = np.array((1,0,0,1,1,0,1,0))
    filament_len = len(genes)

    evo_setup = Setup()
    # should be: evo_setup.FEATURES = (X_train, X_test) ready for sk-learn fit()
    evo_setup.FEATURES = np.random.randint(0,2,(50,filament_len))

    individual = Individual(filament_len = filament_len,
                            genes        = genes)

    print(individual.fitness)

    individual.fitness_eval(evo_setup.FEATURES)

    print(individual.fitness)

    # Check if DNA is binary @TODO: evo for floating points
    print(Individual(filament_len = 8,
                     genes        = (1,0,0,1,1,0,1,0)
                     )
    )

