'''
Shared constants among the program:
User must define othrewise default will be applied
'''

import numpy as np
import os
import datetime as dt


#POP_SIZE = 1000
#MUT_RATE = 0.2
#GENES    = [0,1] # bool[46]
#DNA      = 46
#RHI      = pd.read_csv('./data/rhi_123.csv').to_numpy()
#FEATURES = pd.read_csv('./data/all_features_cleaned.csv').drop(['RHI'],axis='columns').to_numpy()

class Setup(object):
    '''
    Setup class for the evolutionary algorithm experiment.
    
    Parameters (static)
    ---
        POP_SIZE (`static int`): Size of the individuals population
        MUT_RATE (`static float`): Mutation rate @FIXME: should be either a function (e.g., metal cooling).
        GENES(`static list[int]`): list of possible genes, in this case either 0 or 1 for indicating presence/absence.
        FILAMENT_LEN (`static int`): Length of the DNA filament of each `Individual` (e.g., how many features in total)
        FEATURES     (`static tuple(X_train,X_test)`): tuple containing train and test set for fitness evaluation
        LABELS       (`static tuple(y_train,y_test)`): tuple containing train and test set labels for fitness evaualtion
    
    Parameters (constructor)
    ---
        experiment_folder (`str`): (default:`"experiment"`)
        project_prefix    (`str`): path to `experiment_folder/project_prefix + datetime.now()` (default: empty str)
    
    Exmple
    ---
    >>> evo_setup = Setup() #<---- first instanciate
    >>> evo_setup.POP_SIZE = 500 #<---- then init static attributes
    >>> evo_setup.FEATURES = (np.random.randn(50,8),np.random.randn(10,8)) # <---- this should be a tuple of pd.read_csv(...).to_numpy()

    >>> print(evo_setup.POP_SIZE)
    >>> 500
    
    
    '''
    POP_SIZE     : int      = None
    MUT_RATE     : float    = None
    GENES        : list     = [0,1]
    FILAMENT_LEN : int      = None
    DATA         : np.array = None
    LABELS       : np.array = None
    BITS         : dict     = None
    DESCRIPTION  : str      = None
    
    def __init__(self,experiment_folder : str = 'experiment', project_prefix : str = '') -> None:
        self.experiment_folder = experiment_folder
        self.project_folder = os.path.join(experiment_folder,\
                                           project_prefix + dt.datetime\
                                                                .now()\
                                                                    .strftime("%Y%m%d%H%M"))
        
        os.makedirs(self.experiment_folder, exist_ok=True)
        os.makedirs(self.project_folder   , exist_ok=True)





if __name__ == '__main__':
    evo_setup = Setup()#<---- first instanciate
    evo_setup.POP_SIZE = 500 #<---- then init static attributes
    evo_setup.DATA = [1,4,5,6,7,8,9,6,6,4,]

    print(evo_setup.POP_SIZE)