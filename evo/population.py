from evo.individual import Individual

class Population(object):
    def __init__(self) -> None:
        self._population : list[Individual] = []
        self._offspring : list[Individual] = []
    
    @property
    def population(self):
        return self._population
    
    @property
    def offspring(self):
        return self._offspring
    
    def init_population(self,):
        pass
    
    def crossover(self,):
        pass
    
    def selection(self,):
        pass
    
    def mutation(self,):
        pass
    
    def replace(self,):
        for i,o in zip(self._population,self._offspring):
            if i.fitness < o.fitness:
                i = o  
        
        self._population = sorted(self._population, key= lambda x: x.fitness,reverse=True)
        self._offspring = []