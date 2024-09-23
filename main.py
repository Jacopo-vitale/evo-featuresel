from evo.utils import Setup
from evo.individual import Individual
from evo.population import Population
from evo.runner import Runner
import numpy as np


def main():
    setup              = Setup()
    setup.POP_SIZE     = 500
    setup.FILAMENT_LEN = 117
    setup.GENES        = [0,1]
    setup.DATA         = (np.random.randn(50,107), np.random.randn(10,107))
    setup.LABELS       = (np.random.choice([0,1],size=50), np.random.choice([0,1],size=10))
 
    setup.BITS = {
                'features':107,
                'model_selection' : 2,
                'model_param' : 8
            }
    
    pop = Population(setup=setup)
    
    r = Runner(setup=setup,population=pop)
    r.run()





if __name__ == '__main__':
    main()