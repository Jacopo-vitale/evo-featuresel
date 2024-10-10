from evo.individual import Individual
import numpy as np
from evo.utils import Setup
from concurrent.futures import ThreadPoolExecutor, wait
import logging
import sys,os

class Population(object):
    def __init__(self, setup: Setup) -> None:
        self.setup = setup
        self._population : list[Individual] = []
        self._offspring  : list[Individual] = []
        self.logger = logging.getLogger("population")
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

        self.mutation_rate = 1.0
        
    @property
    def population(self):
        return self._population
    
    @property
    def offspring(self):
        return self._offspring
    
    ###############################################################################
    #                                                                             #
    #                          ░▒▓█ POPULATION █▓▒░                               #
    #                                                                             #
    ###############################################################################
    
    def init_individual(self,):
        # New individual
        individual   = \
            Individual(self.setup.FILAMENT_LEN,
                       genes=np.random.choice(self.setup.GENES,size=self.setup.FILAMENT_LEN), 
                       bits=self.setup.BITS,
                       project_folder=self.setup.project_folder)
        
        individual.fitness_eval(self.setup.DATA, self.setup.LABELS)
        return individual
    
    def init_population(self):
        '''
        @FIXME: Introduce all ones Individual --- 25/11 lun,mar,mer 11-13 pick 3 days (topics)
        '''
        with ThreadPoolExecutor() as executor:
            self._population = list(executor.map(lambda _: self.init_individual(),range(self.setup.POP_SIZE)))
        
        self._population = sorted(self._population, key = lambda x: x.fitness,reverse=True)
        
    ###############################################################################
    #                                                                             #
    #                          ░▒▓█ CROSSOVER █▓▒░                                #
    #                                                                             #
    ###############################################################################
    def crossover_individuals(self):
        crossover_point = np.random.randint(1, self.setup.FILAMENT_LEN-1)
        return Individual(filament_len   = self.setup.FILAMENT_LEN,
                          genes          = np.concatenate(
                                                    [np.random.choice(self._population[:int(0.5*self.setup.POP_SIZE)])\
                                                        .genes[:crossover_point],
                                                     np.random.choice(self._population[int(0.5*self.setup.POP_SIZE):])\
                                                        .genes[crossover_point:]]),
                          bits           = self.setup.BITS,
                          project_folder = self.setup.project_folder)
        
        
    def crossover(self,):
        with ThreadPoolExecutor() as executor:
            self._offspring[:] = list(
                executor.map(
                    lambda _: self.crossover_individuals(),range(self.setup.POP_SIZE)
                    )
                )
            
    
    
    ###############################################################################
    #                                                                             #
    #                          ░▒▓█ MUTATION █▓▒░                                 #
    #                                                                             #
    ###############################################################################
    
    def create_mutant(self, individual):
        for i in range(len(individual.genes)):
            if np.random.random() < self.mutation_rate:
                individual.genes[i] = 1 - individual.genes[i]

        individual.fitness_eval(self.setup.DATA, self.setup.LABELS)
        
        return individual
    
    def mutation(self, epoch : int,tot_epoch : int):
        '''
        Using as mutation rate Capacitor discharge formula, increase alpha for increase decading
        '''
        alpha = 0.5
        self.mutation_rate = np.exp(-epoch/(tot_epoch * alpha))
        
        with ThreadPoolExecutor() as executor:
            self._offspring[:] = list(executor.map(self.create_mutant,self._offspring))        

        self._offspring = sorted(self._offspring, key = lambda x: x.fitness,reverse=True)

    ###############################################################################
    #                                                                             #
    #                          ░▒▓█ REPLACING █▓▒░                                #
    #                                                                             #
    ###############################################################################

    def replace(self):
        tmp_generation = []
        for old,new in zip(self._population,self._offspring):
            if old.fitness < new.fitness:
                tmp_generation.append(new)
            else:
                tmp_generation.append(old)
            
        self._population = sorted(tmp_generation, key= lambda x: x.fitness,reverse=True)
        self._offspring.clear()

        self.bestindividual = self._population[0]