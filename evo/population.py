from evo.individual import Individual
import numpy as np
from evo.utils import Setup


class Population(object):
    def __init__(self, setup: Setup) -> None:
        self.setup = setup
        self._population : list[Individual] = []
        self._offspring  : list[Individual] = []
    
    @property
    def population(self):
        return self._population
    
    @property
    def offspring(self):
        return self._offspring
    
    def init_population(self):
        '''
        @FIXME: Introduce all ones Individual --- 25/11 lun,mar,mer 11-13 pick 3 days (topics)
        '''
        for _ in range(self.setup.POP_SIZE):
            genes        = np.random.choice(self.setup.GENES,size=self.setup.FILAMENT_LEN)
            individual   = Individual(self.setup.FILAMENT_LEN,genes=genes, bits=self.setup.BITS,project_folder=self.setup.project_folder)
            individual.fitness_eval(self.setup.DATA, self.setup.LABELS)

            self._population.append(individual)

        self._population = sorted(self._population, key = lambda x: x.fitness,reverse=True)
    
    def crossover(self,):
        
        for _ in range(self.setup.POP_SIZE):
            parent1 = np.random.choice(self._population[:int(0.5*self.setup.POP_SIZE)]) 
            parent2 = np.random.choice(self._population[int(0.5*self.setup.POP_SIZE):]) 

            crossover_point = np.random.randint(1, self.setup.FILAMENT_LEN-1)
            child = Individual(filament_len=self.setup.FILAMENT_LEN,
                               genes = np.concatenate([parent1.genes[:crossover_point],
                                                       parent2.genes[crossover_point:]]),
                                bits=self.setup.BITS,
                                project_folder=self.setup.project_folder)
            
            self._offspring.append(child)

    def mutation(self, epoch : int,tot_epoch : int):
        #alpha ** epoch_counter # formula esponenziale per diminuire mutation rate (diminuisce al variare delle epoche)
        '''
        Using as mutation rate Capacitor discharge formula, increase alpha for increase decading
        '''
        alpha = 0.5
        self.mutation_rate = np.exp(-epoch/(tot_epoch * alpha))
        
        for individual in self._offspring:
            for i in range(len(individual.genes)):
                if np.random.random() < self.mutation_rate:
                    individual.genes[i] = 1 - individual.genes[i]

            individual.fitness_eval(self.setup.DATA, self.setup.LABELS)

        self._offspring = sorted(self._offspring, key = lambda x: x.fitness,reverse=True)


    def replace(self):
        tmp_generation = []
        for old,new in zip(self._population,self._offspring):
            if old.fitness < new.fitness:
                print(f'Replacing {old.fitness} with {new.fitness}')
                tmp_generation.append(new)
            else:
                tmp_generation.append(old)
            
        self._population = sorted(tmp_generation, key= lambda x: x.fitness,reverse=True)
        self._offspring.clear()

        self.bestindividual = self.population[0]