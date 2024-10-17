from evo.utils               import Setup
from evo.population          import Population
from evo.runner              import Runner
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler
import pandas as pd
from joblib import load


def preprocessing(train_subj,test_subj):

    df = pd.read_csv(r'C:\Users\jacop\OneDrive\Desktop\GIST\DATASET\Dataset_features_2D_preproc_quartils_OK.csv')
    X_train = df.query(f'subj  in {train_subj}')[df.columns[:-2]].to_numpy()
    y_train = df.query(f'subj  in {train_subj}')[df.columns[-2]].to_numpy()
    X_test  = df.query(f'subj  in {test_subj}')[df.columns[:-2]].to_numpy()  
    y_test  = df.query(f'subj  in {test_subj}')[df.columns[-2]].to_numpy()

    imputer,scaler = SimpleImputer(),StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test  = scaler.transform(imputer.transform(X_test))
    return ((X_train,X_test),(y_train,y_test))


def main():
    '''
    @FIXME: Rare case is to have genes all ones, maybe better to manually insert
    '''
    list_sub = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    train_subj = [2,3,4,6,7,8,11,12,13,14,15,16,17]
    test_subj = [5,9,10] 
    # Data loading and preproc
    data,labels        = preprocessing(train_subj,test_subj)
    #-------------------------------------------------------------
    setup              = Setup()
    
    setup.POP_SIZE     = 500
    
    setup.BITS = {'features':data[0].shape[1],
                  'model_selection' : 2,
                  'model_params' : 11,
                 }
    
    setup.FILAMENT_LEN = setup.BITS['features'] +\
                         setup.BITS['model_selection']+\
                         setup.BITS['model_params']
    
    setup.GENES        = [0,1]
    setup.DATA         = data
    setup.LABELS       = labels
    setup.RANDOM_SEED  = 42
    setup.DESCRIPTION  = f'pre-processing, 3 subjs test: {test_subj} (2 class 0 and 1 class 1) all the others in train set). Random State equal to {setup.RANDOM_SEED} '
    setup.init_rng()
    #--------------------------------------------------------------
    pop = Population(setup=setup)
    #--------------------------------------------------------------
    r = Runner(setup = setup,population=pop)
    r.run(generations = 200)
    #--------------------------------------------------------------

def load_iron_man():
    iron_man = load(r'experiment\202410161933_ALIEN3\iron_man.joblib')
    print(iron_man)

    return iron_man['preds']

def comparison(preds):

    test_subj = test_subj
    df        = pd.read_csv(r'C:\Users\jacop\OneDrive\Desktop\GIST\DATASET\Dataset_features_2D_preproc_quartils_OK.csv')
    subjs     = df.query(f'subj  in {test_subj}')[df.columns[-1]]  

    _, labels = preprocessing(test_subj)
    y_test    = labels[1]

    results   = {}

    for subj in subjs.unique():
        mask        = subjs == subj
        y_test_subj = y_test[mask]
        preds_subj  = preds[mask]

        correct         = (y_test_subj == preds_subj).sum()
        incorrect       = (y_test_subj != preds_subj).sum()
        total           = len(preds_subj)
        correct_percent = (correct/total) * 100
        error_percent   = (incorrect/total) * 100
    
        results[subj] = {'tot':total, 'correct':correct, 'incorrect':incorrect, 'correct %':correct_percent, 'error %':error_percent}
    print(results)
    #return results


if __name__ == '__main__':
    main()
    #preds = load_iron_man()
    #comparison(preds=preds)


    