from evo.utils import Setup
from evo.individual import Individual
from evo.population import Population
from evo.runner import Runner

def main():
    setup = Setup()
    setup.POP_SIZE = 500
    setup.FILAMENT_LEN = 107 + 2 + 16
    setup.GENES = [0,1]
    





if __name__ == '__main__':
    main()