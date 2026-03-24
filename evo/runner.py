import logging
from evo.population import Population
from evo.utils import Setup
import sys,os
from datetime import datetime
import numpy as np
from joblib import dump

class Runner(object):
    def __init__(self,setup:Setup = None, population : Population = None) -> None:
        self.setup = setup
        self.population = population
        self.logger = logging.getLogger("runner")
        self.logger.setLevel(logging.DEBUG)
        # Create handlers for logging to the standard output and a file
        stdoutHandler = logging.StreamHandler(stream=sys.stdout)
        errHandler = logging.FileHandler(os.path.join(setup.project_folder,"experiment.log"),encoding='utf-8')
        # Set the log levels on the handlers
        stdoutHandler.setLevel(logging.DEBUG)
        errHandler.setLevel(logging.INFO)
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
    
    def step(self,epoch,generations):
        self.logger.info('👪 Starting Crossover 👪')
        self.population.crossover() # qui fa scelta dei genitori con condizioni (ironman e scarpari) e nascita bambini
        self.logger.info(f'Mutant genes rate: {self.population.mutation_rate} ' + np.random.choice(['👽','👾','👹']))
        self.population.mutation(epoch,generations) #mutazione, fitness eval e sort
        self.logger.info('👴👵 Starting Replace 👦👧')
        self.population.replace()
        
    
    def run(self,generations:int = 10,target = 1.0):
        self.welcome()
        self.description(self.setup.DESCRIPTION)
        self.logger.info('🐣 Initializing Population 🐤')
        self.logger.info('###############################################################################')
        self.population.init_population()      # e fitness eval e sortati già
    
        for epoch in range(generations):
            self.logger.info(80*'*')
            self.logger.info(f'Starting epoch {epoch + 1}')
            self.step(epoch,generations)
            self.log_top_five()
            if self.population.bestindividual.fitness >= target:
                self.logger.info(f'Target achieved: {self.population.bestindividual.fitness}')
                self.logger.info(80*'*')
                break
            
        self.logger.info(80*'*')
        self.log_tail()
        
        dump({
              'best_model' : self.population.bestindividual.model,
              'best_genes' : self.population.bestindividual.radiomics,
              'best_fitness':self.population.bestindividual.fitness,
              'best_acc' : self.population.bestindividual.acc,
              'best_f1' : self.population.bestindividual.f1,
              'best_recall' : self.population.bestindividual.recall,
              'best_precision' : self.population.bestindividual.prec,
              'best_cm' : self.population.bestindividual.cm,
              'preds' : self.population.bestindividual.preds

            },
             os.path.join(self.setup.project_folder,'iron_man.joblib'))
    
    
    
    def welcome(self,):
        os.system('cls')
        welcome_message = """
###############################################################################
#                                                                             #
#                      ░▒▓█ WELCOME TO EVO-FEATURESEL █▓▒░                    #
#                                                                             #
#                     🔥🔥🔥        MULTITHREAD       🔥🔥🔥                   #
###############################################################################
        """
        #self.logger.info(' --------------------------------------------------------------')
        #self.logger.info('|                             Welcome                          |')
        #self.logger.info('|                      Evo Feature Selector                    |')
        #self.logger.info('|                                                              |')
        #self.logger.info(' --------------------------------------------------------------')
        self.logger.info(welcome_message)

    def description(self,descr):
        self.logger.info(descr)   
    
    def log_top_five(self,):
        self.logger.info('-------- Top 5 Individuals Fitness --------')
        for i in range(5):
            self.logger.info(f'I{i+1}: ' + f'{self.population.population[i].fitness:0.2f}')
    
    def log_tail(self,):
        self.logger.info('--- Experiment summary ranking ---')
        for i in range(self.setup.POP_SIZE):
            self.logger.info(f'{self.population.population[i].fitness} | Selected: {self.population.population[i].genes.sum()} | Model: {self.population.population[i].model}')
  
    
        

if __name__ == '__main__':
    r = Runner()
    r.run()