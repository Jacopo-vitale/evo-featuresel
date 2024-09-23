import logging
from evo.population import Population

class Runner(object):
    def __init__(self,setup = None,population : Population = None) -> None:
        self.setup = setup
        self.population = population
    
    def step(self):
        print('Starting Crossover...')
        #self.population.crossover()
        print('Starting Mutation...')
        #self.population.mutation()
    
    def run(self,epochs:int = 10,target = 0.9):
        for epoch_counter in range(epochs):
            print(f'Starting epoch {epoch_counter}...')
            self.step()
            print(f'End epoch {epoch_counter}...')


if __name__ == '__main__':
    r = Runner()
    r.run()