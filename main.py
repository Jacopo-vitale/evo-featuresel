from evo.utils import Setup
from evo.individual import Individual
from evo.population import Population
from evo.runner import Runner
import numpy as np


def main():
    setup = Setup()
    setup.POP_SIZE = 500
    setup.FILAMENT_LEN = 110
    setup.GENES = [0,1]
    setup.DATA = (np.random.randn(50,100), np.random.randn(10,100))
    setup.LABELS = (np.random.choice([0,1],size=50), np.random.choice([0,1],size=10))
    
    bits = {
        'features':100,
        'model_selection' : 2,
        'model_param' : 8
    }
    
    i = Individual(filament_len=setup.FILAMENT_LEN,
                   genes=np.random.choice([0,1],size=setup.FILAMENT_LEN),
                   bits=bits,
                   project_folder=setup.project_folder,
                   )
    
    i.fitness_eval(setup.DATA,setup.LABELS)
    print(80*'*')
    r = Runner(setup=setup)
    r.run()





if __name__ == '__main__':
    main()