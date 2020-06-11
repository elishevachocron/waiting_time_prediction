import pandas as pd
from const import *
import numpy as np
import pickle

path_hyperparameters = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Hyperparameters'

dict_algo = {'RF': grid_values_rf , 'SVR': grid_values_svr, 'GBR': grid_values_gbr, 'ANN':grid_values_ann}

for algo in list(model_state.keys()):
    print('--------------------------------------Algo ' + str(algo) + '--------------------------------------')
    df = pd.DataFrame(columns=dict_algo[algo].keys())
    for exp in range(12):
        print('--------------------------------------Experiment ' + str(exp) + '--------------------------------------')
        path_to_load = path + str(exp) + '\\Results'
        for key_type, features_list in features_type.items():
            if key_type == 'set_features_2':
                try:
                    with open(path_to_load + '/'+str(algo)+'_' + key_type + '.pkl', 'rb') as file:  # load the trained model
                        grid_reg = pickle.load(file)
                        estimator = grid_reg.best_estimator_
                        for hyperpara in df.columns:
                            df.loc[exp, hyperpara] = getattr(estimator, hyperpara)
                except:
                    print("A problem occured")
                    continue

    df.to_csv(path_hyperparameters + '/' +str(algo)+'.csv')


