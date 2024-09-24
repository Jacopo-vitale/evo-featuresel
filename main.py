from evo.utils               import Setup
from evo.population          import Population
from evo.runner              import Runner
from sklearn.model_selection import train_test_split
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler
import numpy as np
import pandas as pd


def preprocessing():
    
    df = pd.read_csv('breast_cancer.csv')
    df = df.drop('Sample code number',axis='columns')
    
    data = df[df.columns[:-1]].to_numpy()
    
    classes = df[df.columns[-1]]\
                .apply(lambda x: 0 if x == 2 else 1)\
                    .to_numpy()
    
    imputer,scaler = SimpleImputer(),StandardScaler()
    data = scaler.fit_transform(imputer.fit_transform(data))
    X_train,X_test,y_train,y_test = train_test_split(data,classes)    

    return ((X_train,X_test),(y_train,y_test))


def main():
    '''
    @FIXME: Rare case is to have genes all ones, maybe better to manually insert
    '''
    # Data loading and preproc
    data,labels        = preprocessing()
    
    setup              = Setup()
    setup.POP_SIZE     = 500
    setup.FILAMENT_LEN = 117
    setup.GENES        = [0,1]
    setup.DATA         = data
    setup.LABELS       = labels
 
    setup.BITS = {
                'features':data[0].shape[1],
                'model_selection' : 2,
                'model_param' : 8
            }
    
    pop = Population(setup=setup)
    
    r = Runner(setup=setup,population=pop)
    r.run(epochs=2)





if __name__ == '__main__':
    #preprocessing()
    main()
    