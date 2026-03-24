import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.api import OLS
from statsmodels.regression.linear_model import RegressionResults
import numpy as np
import os

POP_SIZE = 1000
MUT_RATE = 0.2
GENES    = [0,1] # bool[46]
DNA      = 46
RHI      = pd.read_csv('./data/rhi_123.csv').to_numpy()
FEATURES = pd.read_csv('./data/all_features_cleaned.csv').drop(['RHI'],axis='columns').to_numpy()

class Individual(object):
    def __init__(self,DNA,genes) -> None:
        self.genes = genes
        self.fitness = 0.0
        self.DNA = DNA
    
    def set_genes(self,genes):
        self.genes = genes
    
    def eval_fitness(self,):
        if not np.count_nonzero(self.genes):
            print(self.genes)
            raise Exception('All genes are zero... ABORTING')
        
        try:
            selected_features = FEATURES[:,np.array(self.genes,dtype=bool)]
            model = OLS(RHI,selected_features)
            result : RegressionResults = model.fit()
        except:
            raise Exception('Model can\'t be fitted... ABORTING')
        
        self.fitness = result.rsquared

    def __str__(self,) -> str:
        return f'G: {[i for i in self.genes]}\tF: {self.fitness}\n'
    

def initialize_pop() -> list[Individual]:
    population : list[Individual] = []     

    for _ in range(POP_SIZE):
        genes = list(np.random.choice(GENES,size=DNA))
        ind  = Individual(DNA,genes=genes)
        population.append(ind)

    return population

def crossover(selected_chromo : list[Individual], CHROMO_LEN : int, population : list[Individual]) -> list[Individual]:
    offspring_cross : list[Individual] = []

    for _ in range(POP_SIZE):
        parent1 = np.random.choice(selected_chromo)
        parent2 = np.random.choice(population[:int(POP_SIZE*50)])

        crossover_point = np.random.randint(1, CHROMO_LEN-1)
        child = Individual(DNA=DNA,genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:])
        offspring_cross.append(child)

    return offspring_cross

def mutate(offspring : list[Individual], MUT_RATE : float) -> list[Individual]:
    mutated_offspring :list[Individual] = []
    for individual in offspring:
        for i in range(len(individual.genes)):
            if np.random.random() < MUT_RATE:
                individual.genes[i] = 1 - individual.genes[i]
        
        mutated_offspring.append(individual)

    return mutated_offspring

def selection(population : list[Individual]):
    return sorted(population, key = lambda x: x.fitness,reverse=True)[int(0.5*POP_SIZE):]

def threshold(maxIter, generation_):
    thr = 0.8 - float(maxIter / (maxIter + generation_))
    return thr

def fitness_cal(chromo_from_pop : list[Individual]) -> list[Individual]:
    for individual in chromo_from_pop:
        individual.eval_fitness()
    
    return chromo_from_pop

def replace(new_gen :list[Individual], population : list[Individual]) -> list[Individual]:
    for _ in range(len(population)):
        if population[_].fitness < new_gen[_].fitness:
          population[_] = new_gen[_]

    return population

def main(POP_SIZE, MUT_RATE, DNA, GENES):
    # 1) initialize population
    print('Initialising Population...')
    initial_population = initialize_pop()
    found = False
    population : list[Individual] = []
    generation = 1
    max_iter  = 1000
    #threshold = threshold(max_iter,generation)
    threshold = 0.5

    

    # 2) Calculating the fitness for the current population
    print('Calulating individual fitness...')
    for individual in initial_population:
        individual.eval_fitness()
        population.append(individual)
    
    del initial_population

    # now population has 2 things, [chromosome, fitness]
    # 3) now we loop until TARGET is found
    while not found:

      # 3.1) select best people from current population
      print(f'Selecting stronger individuals...POP: {len(population)}')
      selected = selection(population)

      # 3.2) mate parents to make new generation
      print(f'Offspring generation...POP: {len(selected)}')
      population  = sorted(population, key= lambda x:x.fitness,reverse=True)
      crossovered = crossover(selected, DNA, population)
            
      # 3.3) mutating the childeren to diversfy the new generation
      print(f'Offspring mutation...POP: {len(crossovered)}')
      mutated = mutate(crossovered, MUT_RATE)

      new_gen = []
      for mutant in mutated:
        mutant.eval_fitness()
        new_gen.append(mutant)

      # 3.4) replacement of bad population with new generation
      # we sort here first to compare the least fit population with the most fit new_gen
      print(f'Replacing with stronger individuals...POP: {len(new_gen)}')
      population = replace(new_gen, population)

      

      del new_gen
 
      if ((population[0].fitness >= threshold) or (generation == max_iter)):
        os.system('cls')
        print('Target found')
        print('String: ' + str(population[0].genes) + ' Generation: ' + str(generation) + ' Fitness: ' + str(population[0].fitness))
        break
      
      os.system('cls')
      print('Generation: ' + str(generation) + ' Fitness: ' + str(population[0].fitness))
      print('Generation: ' + str(generation) + ' Fitness: ' + str(population[1].fitness))
      print('Generation: ' + str(generation) + ' Fitness: ' + str(population[2].fitness))
      generation+=1
    



main(POP_SIZE, MUT_RATE,DNA, GENES)