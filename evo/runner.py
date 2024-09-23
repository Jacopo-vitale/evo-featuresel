import logging
from evo.population import Population
from evo.utils import Setup
import sys,os

class Runner(object):
    def __init__(self,setup:Setup = None, population : Population = None) -> None:
        self.setup = setup
        self.population = population
        self.logger = logging.getLogger("runner")
        self.logger.setLevel(logging.DEBUG)
        # Create handlers for logging to the standard output and a file
        stdoutHandler = logging.StreamHandler(stream=sys.stdout)
        errHandler = logging.FileHandler(os.path.join(setup.project_folder,"error.log"))
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
    
    def step(self,epoch_counter):
        self.logger.info('Starting Crossover...')
        self.population.crossover() # qui fa scelta dei genitori con condizioni (ironman e scarpari) e nascita bambini
        self.logger.info('Starting Mutation...')
        self.population.mutation(epoch_counter) #mutazione, fitness eval e sort
        self.logger.info('Starting Replace...')
        self.population.replace()
        
    
    def run(self,epochs:int = 10,target = 0.9):
        self.logger.info('Initializing Population...')
        self.population.init_population()      # e fitness eval e sortati già
    

        for epoch_counter in range(epochs):
            self.logger.info(f'Starting epoch {epoch_counter}...')
            self.step(epoch_counter)
            if self.population.bestindividual.fitness >= target:
                self.logger.info(f'Target achieved: {target}')
                break
            self.logger.info(f'End epoch {epoch_counter}...')
            self.logger.info('**************************')
            self.logger.info(f'Best fitness is: {self.population.bestindividual.fitness} ')


if __name__ == '__main__':
    r = Runner()
    r.run()