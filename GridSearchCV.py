import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import math
import warnings
import time
import numpy as np
from Data_preprocessing import X_train, X_test, y_train, y_test, y_HOL_test, y_LES_test, y_QTD_test
start = time.time()
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------- #
# ---------------------- PARAMETERS --------------------------------- #
# ------------------------------------------------------------------- #
algorithms_used = ['RF', 'SVR', 'GBR', 'ANN', 'QTD', 'HOL']
n_folds = 5

# ---------------------- Random Forest ------------------------------- #
grid_values_rf = {'max_depth': [1, 3, 10, 15], 'n_estimators': [100, 150, 200, 250]}

# ---------------------------- SVR ---------------------------------- #
grid_values_svr = {'kernel': ('linear', 'rbf'), 'C': [0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25],\
                   'max_iter': [100, 500, 1000]}

# ---------------------------- GBR ---------------------------------- #
grid_values_gbr = {'loss': ['ls', 'lad'], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [10, 50, 150]}

# ----------------------------- ANN ----------------------------------- #
grid_values_ann = {'hidden_layer_sizes': [5, 30, 80],\
                 'activation': ['identity', 'logistic', 'relu'], 'learning_rate': ['invscaling', 'adaptive']}

# -------- Cross-fold ----------------------------

features_type = {'features_type_1': ['Arrival_time', 'n_servers', 'LSD', 'queue1', 'queue2', 'Day']}
np.random.seed(42)

for key_type, features_list in features_type.items():
    print('-------------------------------------- Type_features: ' + str(
            key_type) + ' ----------------------------------')
        # results_dict = {1: {}, 3: {}, 5: {}, 7: {}, 9: {}, 10: {}}
    results_dict = {'RF': {}, 'SVR': {}, 'GBR': {}, 'ANN': {}, 'QTD': {}, 'LES': {}, 'HOL': {}}

    path = 'C:\\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Synthetic_Data_Generation\Good Split'
    path_to_save = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Synthetic_Data_Generation\GridSearchCV\Good_Split'

    # X_train = pd.read_csv(path + '\\X_train.csv')
    # X_test = pd.read_csv(path + '\\X_test.csv')
    # y_train = pd.read_csv(path + '\\y_train.csv')
    # y_test = pd.read_csv(path + '\\y_test.csv')
    # y_QTD_test = pd.read_csv(path + '\\y_QTD_test.csv')
    # y_LES_test = pd.read_csv(path + '\\y_LES_test.csv')
    # y_HOL_test = pd.read_csv(path + '\\y_HOL_test.csv')
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test.reset_index(drop=True, inplace=True)
    y_QTD_test.reset_index(drop=True, inplace=True)
    y_LES_test.reset_index(drop=True, inplace=True)
    y_HOL_test.reset_index(drop=True, inplace=True)
    mean_WT_test = float(y_test.mean() * 3600)
    # non Zero label
    y_test_not_zero = np.array(y_test.loc[y_test.Real_WT != 0]).reshape(-1)  # y that are not zero
    y_QTD_test_not_zero = y_QTD_test.to_numpy()[y_test.loc[y_test.Real_WT != 0].index].reshape(-1)
    y_LES_test_not_zero = y_LES_test.to_numpy()[y_test.loc[y_test.Real_WT != 0].index].reshape(-1)
    y_HOL_test_not_zero = y_HOL_test.to_numpy()[y_test.loc[y_test.Real_WT != 0].index].reshape(-1)

    for algo in algorithms_used:
        if algo == 'RF':
            start_rf = time.time()
            rf = RandomForestRegressor()
            grid_rf = GridSearchCV(rf, param_grid=grid_values_rf, scoring='neg_mean_squared_error')
            print('---------------RF FITTING---------------')
            grid_rf.fit(X_train, y_train)
            # with open(path_to_save + '/RF.pkl', 'rb') as file:
            #     grid_rf = pickle.load(file)
            y_pred_rf = grid_rf.predict(X_test)
            with open(path_to_save + '/RF_without_LES_7.pkl', 'wb') as file: #save the trained model
                pickle.dump(grid_rf, file)
            y_pred_not_zero = np.array(y_pred_rf[y_test.loc[y_test.Real_WT != 0].index])
            re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()

            rmse = math.sqrt(mean_squared_error(y_test, y_pred_rf)) * 3600
            mae = mean_absolute_error(y_test, y_pred_rf)*3600
            cv = rmse/mean_WT_test
            results_dict['RF'] = [grid_rf.best_estimator_, rmse, mae, cv, re]
            end_rf = time.time()
            print('Time elapsed RF: ', int((end_rf-start_rf)/60))
        elif algo == 'SVR':
            start_svr = time.time()
            svr = SVR()
            grid_svr = GridSearchCV(svr, param_grid=grid_values_svr, scoring='neg_mean_squared_error')
            print('---------------SVR FITTING---------------')
            grid_svr.fit(X_train, y_train)
            # with open(path_to_save + '/SVR.pkl', 'rb') as file:
            #     grid_svr = pickle.load(file)
            y_pred_svr = grid_svr.predict(X_test)
            with open(path_to_save + '/SVR_without_LES_7.pkl', 'wb') as file: #save the trained model
                pickle.dump(grid_svr, file)
            y_pred_not_zero = np.array(y_pred_svr[y_test.loc[y_test.Real_WT != 0].index])
            re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
            rmse = math.sqrt(mean_squared_error(y_test, y_pred_svr)) * 3600
            mae = mean_absolute_error(y_test, y_pred_svr) * 3600
            cv = rmse / mean_WT_test
            results_dict['SVR'] = [grid_svr.best_estimator_, rmse, mae, cv, re]
            end_svr = time.time()
            print('Time elapsed SVR: ', int((end_svr-start_svr)/60))

        elif algo == 'GBR':
            start_gbr = time.time()
            gbr = GradientBoostingRegressor()
            grid_gbr = GridSearchCV(gbr, param_grid=grid_values_gbr, scoring='neg_mean_squared_error')
            print('---------------GBR FITTING---------------')
            grid_gbr.fit(X_train, y_train)
            # with open(path_to_save + '/GBR.pkl', 'rb') as file:
            #     grid_gbr = pickle.load(file)
            y_pred_gbr = grid_gbr.predict(X_test)
            with open(path_to_save + '/GBR_without_LES_7.pkl', 'wb') as file: #save the trained model
                pickle.dump(grid_gbr, file)
            y_pred_not_zero = np.array(y_pred_gbr[y_test.loc[y_test.Real_WT != 0].index])
            re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
            rmse = math.sqrt(mean_squared_error(y_test, y_pred_gbr)) * 3600
            mae = mean_absolute_error(y_test, y_pred_gbr) * 3600
            cv = rmse / mean_WT_test
            results_dict['GBR'] = [grid_gbr.best_estimator_, rmse, mae, cv, re]
            end_gbr = time.time()
            print('Time elapsed GBR: ', int((end_gbr-start_gbr)/60))

        elif algo == 'ANN':
            start_ann = time.time()
            ann = MLPRegressor()
            grid_ann = GridSearchCV(ann, param_grid=grid_values_ann, scoring='neg_mean_squared_error')
            print('---------------ANN FITTING---------------')
            grid_ann.fit(X_train, y_train)
            # with open(path_to_save + '/ANN.pkl', 'rb') as file:
            #     grid_ann = pickle.load(file)
            y_pred_ann = grid_ann.predict(X_test)
            with open(path_to_save + '/ANN_without_LES_7.pkl', 'wb') as file: #save the trained model
                pickle.dump(grid_ann, file)
            y_pred_not_zero = np.array(y_pred_ann[y_test.loc[y_test.Real_WT != 0].index])
            re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
            rmse = math.sqrt(mean_squared_error(y_test, y_pred_ann)) * 3600
            mae = mean_absolute_error(y_test, y_pred_ann) * 3600
            cv = rmse / mean_WT_test
            results_dict['ANN'] = [grid_ann.best_estimator_, rmse, mae, cv, re]
            end_ann = time.time()
            print('Time elapsed ANN: ', int((end_ann-start_ann)/60))

        elif algo == 'QTD':
            print('---------------QTD Prediction---------------')
            re = (np.divide(np.absolute(y_test_not_zero - y_QTD_test_not_zero), y_test_not_zero)).mean()
            mae = mean_absolute_error(y_test, y_QTD_test) * 3600
            rmse = math.sqrt(mean_squared_error(y_test, y_QTD_test)) * 3600
            cv = rmse / mean_WT_test
            results_dict['QTD'] = [None, rmse, mae, cv, re]

        elif algo == 'LES':
            print('---------------LES Prediction---------------')
            re = (np.divide(np.absolute(y_test_not_zero - y_LES_test_not_zero), y_test_not_zero)).mean()
            mae = mean_absolute_error(y_test, y_LES_test) * 3600
            rmse = math.sqrt(mean_squared_error(y_test, y_LES_test)) * 3600
            cv = rmse / mean_WT_test
            results_dict['LES'] = [None, rmse, mae, cv, re]

        elif algo == 'HOL':
            print('---------------HOL Prediction---------------')
            re = (np.divide(np.absolute(y_test_not_zero - y_HOL_test_not_zero), y_test_not_zero)).mean()
            mae = mean_absolute_error(y_test, y_HOL_test) * 3600
            rmse = math.sqrt(mean_squared_error(y_test, y_HOL_test)) * 3600
            cv = rmse / mean_WT_test
            results_dict['HOL'] = [None, rmse, mae, cv, re]


    with open(path_to_save+ '/results_dict_'+ str(key_type)+'.p', 'wb') as fp:
        pickle.dump(results_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    rmse_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    mae_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    cv_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    re_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    hyper_parameters = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    labels = []

    df_results = pd.DataFrame(columns=['RMSE', 'MAE', 'CV', 'RE'],
                              index=['RF', 'SVR', 'GBR', 'ANN', 'QTD',  'HOL']) #to add 'HyperParameters' to column and LES to index

    for algo, results in results_dict.items():
        #hyper_parameters[algo].append(results[0])
        if algo!='LES':
            rmse_metrics[algo].append(results[1])
            mae_metrics[algo].append(results[2])
            cv_metrics[algo].append(results[3])
            re_metrics[algo].append(results[4])

    df_results['RMSE'] = np.concatenate(list(rmse_metrics.values()))
    df_results['MAE'] = np.concatenate(list(mae_metrics.values()))
    df_results['CV'] = np.concatenate(list(cv_metrics.values()))
    df_results['RE'] = np.concatenate(list(re_metrics.values()))
    #df_results['HyperParameters'] = list(hyper_parameters.values())

    df_results.to_csv(
        r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Synthetic_Data_Generation\GridSearchCV\Good_Split/Results_without_LES_7.csv')

end = time.time()
print('Time elapsed: ', int((end-start)/60))

