import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
#import xgboost as xgb
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import math
import warnings
import time
import numpy as np
from const import *

start = time.time()
warnings.filterwarnings('ignore')

np.random.seed(42)

#for experiment in np.arange(3, 4, 1):
for experiment in [3, 5, 6, 7, 8]:
    print('--------------------------------------Experiment ' + str(experiment)+'--------------------------------------')
    path_to_load = path + str(experiment)
    path_to_save = path + str(experiment) + '\\Results_new_features_normalized'
    for key_type, features_list in features_type.items():
        print('--------------------------------------Data Pre-processing----------------------------------')
        print('-------------------------------------- Set_features: ' + str(key_type) + ' ----------------------------------')

        results_dict = {'RF': {}, 'SVR': {}, 'GBR': {}, 'XGBR': {}, 'ANN': {}, 'QTD': {}, 'LES': {}, 'HOL': {}}

        data = pd.read_csv(path_to_load+'/New_features_simulation.csv')

        data.loc[data.Real_WT == np.inf, 'Real_WT'] = data.loc[data.Real_WT == np.inf, 'Exit_time'] \
                                                      - data.loc[data.Real_WT == np.inf, 'Arrival_time']  # replace the abandonment cases

        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Real_WT'], axis=0, how="any")# remove all the missing value
        #TODO check if ther is any missing value (data[features_list].isna().sum())
        df = data[columns_list]
        df = data
    # ---------------Test/Train Split----------------
        print('Test/Train Split')
        # QT estimation of Waiting Time
        week_split = 4 #6 for 9 weeks simulation and 4 for 6 weeks simulation
        y_QTD_test = df.loc[df.Week > week_split]['WT_QTD_bis'].to_frame()
        y_QTD_test.reset_index(drop=True, inplace=True)

        y_LES_test = df.loc[df.Week > week_split]['LES'].to_frame()
        y_LES_test.reset_index(drop=True, inplace=True)
        y_HOL_test = df.loc[df.Week > week_split]['HOL'].to_frame()
        y_HOL_test.reset_index(drop=True, inplace=True)
        # Real Waiting Time
        y = df['Real_WT'].to_frame()
        X_train = df.loc[df.Week <= week_split][features_list]
        X_test = df.loc[df.Week > week_split][features_list]
        X_test.reset_index(drop=True, inplace=True)
        y_train = df.loc[df.Week <= week_split]['Real_WT'].to_frame()
        y_test = df.loc[df.Week > week_split]['Real_WT'].to_frame()
        y_test.reset_index(drop=True, inplace=True)
        mean_WT_test = float(y_test.mean() * 3600)
        # non Zero label(higher than 2 seconds)
        y_test_not_zero = np.array(y_test.loc[y_test.Real_WT > (2 / 3600)]).reshape(-1)  # y that are not zero
        y_QTD_test_not_zero = y_QTD_test.to_numpy()[y_test.loc[y_test.Real_WT > (2 / 3600)].index].reshape(-1)
        y_LES_test_not_zero = y_LES_test.to_numpy()[y_test.loc[y_test.Real_WT > (2 / 3600)].index].reshape(-1)
        y_HOL_test_not_zero = y_HOL_test.to_numpy()[y_test.loc[y_test.Real_WT > (2 / 3600)].index].reshape(-1)

        # Normalization
        X_train_norm = (X_train-X_train.min())/(X_train.max()-X_train.min())
        X_test_norm = (X_test-X_train.min())/(X_train.max()-X_train.min())

        # Stantardization
        X_train_stand = (X_train-X_train.mean())/(X_train.std())
        X_test_stand = (X_test-X_train.mean())/(X_train.std())



        for algo in algorithms_used:
            if algo == 'RF':
                if model_state[algo] == 'fit':
                    start_rf = time.time()
                    print('---------------RF FITTING---------------')
                    rf = RandomForestRegressor()
                    grid_rf = GridSearchCV(rf, param_grid=grid_values_rf, scoring='neg_mean_squared_error')
                    grid_rf.fit(X_train_norm, y_train)
                    end_rf = time.time()
                    print('Time elapsed RF: ', int((end_rf - start_rf) / 60))
                    with open(path_to_save + '/RF_'+key_type+'.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_rf, file)
                elif model_state[algo] == 'saved':
                    print('---------------RF LOADING---------------')
                    with open(path_to_save + '/RF_'+key_type+'.pkl', 'rb') as file: #load the trained model
                        grid_rf = pickle.load(file)
                y_pred_rf = grid_rf.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_rf[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
                rmse = math.sqrt(mean_squared_error(y_test, y_pred_rf)) * 3600
                mae = mean_absolute_error(y_test, y_pred_rf)*3600
                cv = rmse/mean_WT_test
                results_dict['RF'] = [grid_rf.best_estimator_, rmse, mae, cv, re]
            elif algo == 'SVR':
                if model_state[algo] == 'fit':
                    start_svr = time.time()
                    print('---------------SVR FITTING---------------')
                    svr = SVR()
                    grid_svr = GridSearchCV(svr, param_grid=grid_values_svr, scoring='neg_mean_squared_error')
                    grid_svr.fit(X_train_stand, y_train)
                    end_svr = time.time()
                    print('Time elapsed SVR: ', int((end_svr - start_svr) / 60))
                    with open(path_to_save + '/SVR_'+key_type+'.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_svr, file)
                elif model_state[algo] == 'saved':
                    print('---------------SVR LOADING---------------')
                    with open(path_to_save + '/SVR_'+key_type+'.pkl', 'rb') as file: #load the trained model
                        grid_svr = pickle.load(file)
                y_pred_svr = grid_svr.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_svr[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
                rmse = math.sqrt(mean_squared_error(y_test, y_pred_svr)) * 3600
                mae = mean_absolute_error(y_test, y_pred_svr) * 3600
                cv = rmse / mean_WT_test
                results_dict['SVR'] = [grid_svr.best_estimator_, rmse, mae, cv, re]

            # elif algo == 'XGBR':
            #     if model_state[algo] == 'fit':
            #         start_xgbr = time.time()
            #         print('---------------XGBR FITTING---------------')
            #         xgbr = xgb.XGBRegressor()
            #         grid_xgbr = GridSearchCV(xgbr, param_grid=grid_values_xgbr, scoring='neg_mean_squared_error') #TODO check if the paramgrid gbr = xbr
            #         grid_xgbr.fit(X_train, y_train)
            #         end_xgbr = time.time()
            #         print('Time elapsed XGBR: ', int((end_xgbr - start_xgbr) / 60))
            #         with open(path_to_save + '/XGBR_' + key_type + '.pkl', 'wb') as file:  # save the trained model
            #             pickle.dump(grid_xgbr, file)
            #     elif model_state[algo] == 'saved':
            #         print('---------------XGBR LOADING---------------')
            #         with open(path_to_save + '/XGBR_' + key_type + '.pkl', 'rb') as file:  # load the trained model
            #             grid_xgbr = pickle.load(file)
            #     y_pred_xgbr = grid_gbr.predict(X_test)
            #     y_pred_not_zero = np.array(y_pred_xgbr[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
            #     re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
            #     rmse = math.sqrt(mean_squared_error(y_test, y_pred_xgbr)) * 3600
            #     mae = mean_absolute_error(y_test, y_pred_xgbr) * 3600
            #     cv = rmse / mean_WT_test
            #     results_dict['XGBR'] = [grid_xgbr.best_estimator_, rmse, mae, cv, re]

            elif algo == 'GBR':
                if model_state[algo] == 'fit':
                    start_gbr = time.time()
                    print('---------------GBR FITTING---------------')
                    gbr = GradientBoostingRegressor()
                    grid_gbr = GridSearchCV(gbr, param_grid=grid_values_gbr, scoring='neg_mean_squared_error')
                    grid_gbr.fit(X_train_norm, y_train)
                    end_gbr = time.time()
                    print('Time elapsed GBR: ', int((end_gbr - start_gbr) / 60))
                    with open(path_to_save + '/GBR_'+key_type+'.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_gbr, file)
                elif model_state[algo] == 'saved':
                    print('---------------GBR LOADING---------------')
                    with open(path_to_save + '/GBR_'+key_type+'.pkl', 'rb') as file: #load the trained model
                        grid_gbr = pickle.load(file)
                y_pred_gbr = grid_gbr.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_gbr[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
                rmse = math.sqrt(mean_squared_error(y_test, y_pred_gbr)) * 3600
                mae = mean_absolute_error(y_test, y_pred_gbr) * 3600
                cv = rmse / mean_WT_test
                results_dict['GBR'] = [grid_gbr.best_estimator_, rmse, mae, cv, re]

            elif algo == 'ANN':
                if model_state[algo] == 'fit':
                    start_ann = time.time()
                    print('---------------ANN FITTING---------------')
                    ann = MLPRegressor()
                    grid_ann = GridSearchCV(ann, param_grid=grid_values_ann, scoring='neg_mean_squared_error')
                    grid_ann.fit(X_train_norm, y_train)
                    end_ann = time.time()
                    print('Time elapsed ANN: ', int((end_ann - start_ann) / 60))
                    with open(path_to_save + '/ANN_'+key_type+'.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_ann, file)
                elif model_state[algo] == 'saved':
                    print('---------------ANN LOADING---------------')
                    with open(path_to_save + '/ANN_'+key_type+'.pkl', 'rb') as file: #load the trained model
                        grid_ann = pickle.load(file)
                y_pred_ann = grid_ann.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_ann[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
                rmse = math.sqrt(mean_squared_error(y_test, y_pred_ann)) * 3600
                mae = mean_absolute_error(y_test, y_pred_ann) * 3600
                cv = rmse / mean_WT_test
                results_dict['ANN'] = [grid_ann.best_estimator_, rmse, mae, cv, re]

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


        with open(path_to_save+ '/results_dict_'+ key_type+'.p', 'wb') as fp:
            pickle.dump(results_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        rmse_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'XGBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        mae_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'XGBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        cv_metrics = {'RF': [], 'SVR': [], 'GBR': [],  'XGBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        re_metrics = {'RF': [], 'SVR': [], 'GBR': [],  'XGBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        hyper_parameters = {'RF': [], 'SVR': [], 'GBR': [], 'XGBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        labels = []

        df_results = pd.DataFrame(columns=['RMSE', 'MAE', 'CV', 'RE', 'HyperParameters'],
                                  index=['RF', 'SVR', 'GBR', 'XGBR', 'ANN', 'QTD', 'LES', 'HOL'])

        for algo, results in results_dict.items():
            hyper_parameters[algo].append(results[0])
            rmse_metrics[algo].append(results[1])
            mae_metrics[algo].append(results[2])
            cv_metrics[algo].append(results[3])
            re_metrics[algo].append(results[4])

        df_results['RMSE'] = np.concatenate(list(rmse_metrics.values()))
        df_results['MAE'] = np.concatenate(list(mae_metrics.values()))
        df_results['CV'] = np.concatenate(list(cv_metrics.values()))
        df_results['RE'] = np.concatenate(list(re_metrics.values()))
        df_results['HyperParameters'] = list(hyper_parameters.values())

        df_results.to_csv(path_to_save+'/Results_'+key_type+'.csv')

    end = time.time()
    print('Time elapsed: ', int((end-start)/60))

