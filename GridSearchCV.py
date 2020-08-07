import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
#import xgboost as xgb
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import math
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from const import *
import warnings


warnings.filterwarnings('ignore', 'Solver terminated early.*')

start = time.time()
warnings.filterwarnings('ignore')

np.random.seed(42)
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for experiment in [9, 10]:
    print('--------------------------------------Experiment ' + str(experiment)+'--------------------------------------')
    path_to_load = path + str(experiment)
    path_to_save = path + str(experiment) + '\\Results'
    for key_type, features_list in features_type.items():
        if key_type == 'set_features_2':
            continue
        print('--------------------------------------Data Pre-processing----------------------------------')
        print('-------------------------------------- Set_features: ' + str(key_type) + ' ----------------------------------')

        results_dict = {'RF': {}, 'SVR': {}, 'GBR': {}, 'ANN': {}, 'LR': {}, 'QTD': {}, 'LES': {}, 'HOL': {}}
        data = pd.read_csv(path_to_load+'/New_features_simulation.csv')

        data.loc[data.Real_WT == np.inf, 'Real_WT'] = data.loc[data.Real_WT == np.inf, 'Exit_time'] \
                                                      - data.loc[data.Real_WT == np.inf, 'Arrival_time'] # replace the abandonment cases
        if experiment == 0:#removing the night's hours
            data = data.loc[(data.Arrival_time >= 8) & (data.Arrival_time < 20)]
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Real_WT'], axis=0, how="any")# remove all the missing value
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Checkin_time'], axis=0, how="any") #remove on th abandonment cases
        # data = data.loc[data.Checkin_time.isna()] #learn on th abandonment cases
        # print('Number of samples: ', len(data))
        #TODO check if ther is any missing value (data[features_list].isna().sum())

        if (data[features_list].isna().sum() == 1).any():
            print('MISSING VALUES')

        df = data
        if experiment in [1, 2, 4, 5, 6, 7, 8]:  # experiment without abandonment
            final_features_list = np.delete(np.array(features_list), 8)# to avoid NaN after normalizing
            #final_features_list = np.delete(np.array(features_list), 9)
        else:
            final_features_list = features_list

    # ---------------Test/Train Split----------------
        print('Test/Train Split')
        # QT estimation of Waiting Time
        week_split = 4 #6 for 9 weeks simulation, 4 for 6 weeks simulation, 24 for 30 weeks simulation split, inverse split = 2 + change the sign of the test and train
        y_QTD_test = df.loc[df.Week > week_split]['WT_QTD_bis'].to_frame()
        y_QTD_test.reset_index(drop=True, inplace=True)

        y_LES_test = df.loc[df.Week > week_split]['LES'].to_frame()
        y_LES_test.reset_index(drop=True, inplace=True)
        # TODO check if ther is any missing value (y_LES_test.isna().sum())
        y_HOL_test = df.loc[df.Week > week_split]['HOL'].to_frame()
        y_HOL_test.reset_index(drop=True, inplace=True)
        # Real Waiting Time
        y = df['Real_WT'].to_frame()
        X_train = df.loc[df.Week <= week_split][final_features_list]
        X_test = df.loc[df.Week > week_split][final_features_list]
        X_test.reset_index(drop=True, inplace=True)
        y_train = df.loc[df.Week <= week_split]['Real_WT'].to_frame()
        #y_test = df.loc[df.Week > week_split]['Real_WT'].to_frame()
        y_test = df.loc[df.Week > week_split]['Real_WT'].to_frame()
        y_test.reset_index(drop=True, inplace=True)
        mean_WT_train = float(y_train.mean() * 3600)
        mean_WT_test = float(y_test.mean() * 3600)
        # non Zero label(higher than 2 seconds)
        y_test_not_zero = np.array(y_test.loc[y_test.Real_WT > (2 / 3600)]).reshape(-1)  # y that are not zero
        y_QTD_test_not_zero = y_QTD_test.to_numpy()[y_test.loc[y_test.Real_WT > (2 / 3600)].index].reshape(-1)
        y_LES_test_not_zero = y_LES_test.to_numpy()[y_test.loc[y_test.Real_WT > (2 / 3600)].index].reshape(-1)
        y_HOL_test_not_zero = y_HOL_test.to_numpy()[y_test.loc[y_test.Real_WT > (2 / 3600)].index].reshape(-1)

        # Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)


        for algo in algorithms_used:
            if algo == 'RF':
                if model_state[algo] == 'fit':
                    start_rf = time.time()
                    print('---------------RF FITTING---------------')
                    rf = RandomForestRegressor(oob_score=True)
                    grid_rf = GridSearchCV(rf, param_grid=grid_values_rf, scoring='neg_mean_squared_error', n_jobs=5)
                    grid_rf.fit(X_train_norm, y_train.to_numpy().reshape(-1))
                    end_rf = time.time()
                    print('Time elapsed RF: ', int((end_rf - start_rf) / 60))
                    with open(path_to_save + '/RF_'+key_type+'.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_rf, file)
                    with open(path_to_save + '/Computation_time_RF.txt', 'a+') as f: #save the computation time
                        f.write(' Computation time: ' + str(int((end_rf - start_rf) / 60)) + ' minutes')
                elif model_state[algo] == 'saved':
                    print('---------------RF LOADING---------------')
                    with open(path_to_save + '/RF_'+key_type+'.pkl', 'rb') as file: #load the trained model
                        grid_rf = pickle.load(file)

                #y_pred_rf_train = grid_rf.best_estimator_.oob_prediction_
                y_pred_rf_train = grid_rf.predict(X_train_norm)
                y_pred_rf_test = grid_rf.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_rf_test[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
                mae = mean_absolute_error(y_test, y_pred_rf_test) * 3600
                rmae = mae / mean_WT_test
                rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_rf_train)) * 3600
                rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_rf_test)) * 3600
                rrmse_train = rmse_train / mean_WT_train
                rrmse_test = rmse_test/mean_WT_test
                bias = np.mean([y_test.to_numpy()[i] - y_pred_rf_test[i] for i in range(len(y_test))])*3600
                results_dict['RF'] = [grid_rf.best_estimator_, round(rmse_train, 2), round(rmse_test, 2), round(rrmse_train, 2),
                                      round(rrmse_test, 2), round(rmae, 2), round(mae, 2), round(re, 2), round(bias, 2)]
                # plt.figure()
                # plt.bar(np.arange(len(final_features_list)), grid_rf.best_estimator_.feature_importances_)
                # plt.xticks(np.arange(len(final_features_list)), final_features_list)
                # plt.savefig(path_to_save + '\\Feature_importance/' + algo + '.png')

            elif algo == 'SVR':
                if model_state[algo] == 'fit':
                    start_svr = time.time()
                    print('---------------SVR FITTING---------------')
                    svr = SVR()
                    grid_svr = GridSearchCV(svr, param_grid=grid_values_svr, scoring='neg_mean_squared_error', n_jobs=5)
                    grid_svr.fit(X_train_norm, y_train.to_numpy().reshape(-1))
                    end_svr = time.time()
                    print('Time elapsed SVR: ', int((end_svr - start_svr) / 60))
                    with open(path_to_save + '/SVR_'+key_type+'.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_svr, file)
                    with open(path_to_save + '/Computation_time_SVR_4.txt', 'a+') as f: #save the computation time
                        f.write('Computation time: ' + str(int((end_svr - start_svr) / 60)) + ' minutes')
                elif model_state[algo] == 'saved':
                    print('---------------SVR LOADING---------------')
                    with open(path_to_save + '/SVR_'+key_type+'.pkl', 'rb') as file: #load the trained model
                        grid_svr = pickle.load(file)
                y_pred_svr_train = grid_svr.predict(X_train_norm)
                y_pred_svr_test = grid_svr.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_svr_test[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
                mae = mean_absolute_error(y_test, y_pred_svr_test) * 3600
                rmae = mae / mean_WT_test
                rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_svr_train)) * 3600
                rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_svr_test)) * 3600
                rrmse_train = rmse_train / mean_WT_train
                rrmse_test = rmse_test / mean_WT_test
                bias = np.mean([y_test.to_numpy()[i] - y_pred_svr_test[i] for i in range(len(y_test))]) * 3600
                results_dict['SVR'] = [grid_svr.best_estimator_, round(rmse_train, 2), round(rmse_test, 2), round(rrmse_train, 2),
                                      round(rrmse_test, 2), round(rmae, 2), round(mae, 2), round(re, 2), round(bias, 2)]

            elif algo == 'GBR':
                if model_state[algo] == 'fit':
                    start_gbr = time.time()
                    print('---------------GBR FITTING---------------')
                    gbr = GradientBoostingRegressor()
                    grid_gbr = GridSearchCV(gbr, param_grid=grid_values_gbr, scoring='neg_mean_squared_error', n_jobs=8)
                    grid_gbr.fit(X_train_norm, y_train.to_numpy().reshape(-1))
                    end_gbr = time.time()
                    print('Time elapsed GBR: ', int((end_gbr - start_gbr) / 60))
                    with open(path_to_save + '/GBR_'+key_type+'.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_gbr, file)
                    with open(path_to_save + '/Computation_time_GBR_4.txt', 'a+') as f:  # save the computation time
                        f.write('Computation time: ' + str(int((end_gbr - start_gbr) / 60)) + ' minutes')
                elif model_state[algo] == 'saved':
                    print('---------------GBR LOADING---------------')
                    with open(path_to_save + '/GBR_'+key_type+'.pkl', 'rb') as file: #load the trained model
                        grid_gbr = pickle.load(file)
                y_pred_gbr_train = grid_gbr.predict(X_train_norm)
                y_pred_gbr_test = grid_gbr.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_gbr_test[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
                mae = mean_absolute_error(y_test, y_pred_gbr_test) * 3600
                rmae = mae / mean_WT_test
                rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_gbr_train)) * 3600
                rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_gbr_test)) * 3600
                rrmse_train = rmse_train / mean_WT_train
                rrmse_test = rmse_test / mean_WT_test
                bias = np.mean([y_test.to_numpy()[i] - y_pred_gbr_test[i] for i in range(len(y_test))]) * 3600
                results_dict['GBR'] = [grid_gbr.best_estimator_, round(rmse_train, 2), round(rmse_test, 2), round(rrmse_train, 2),
                                      round(rrmse_test, 2), round(rmae, 2), round(mae, 2), round(re, 2), round(bias, 2)]
                # plt.figure()
                # plt.bar(np.arange(len(final_features_list)), grid_gbr.best_estimator_.feature_importances_)
                # plt.xticks(np.arange(len(final_features_list)), final_features_list)
                # plt.savefig(path_to_save + '\\Feature_importance/' + algo + '.png')

            elif algo == 'ANN':
                if model_state[algo] == 'fit':
                    start_ann = time.time()
                    print('---------------ANN FITTING---------------')
                    ann = MLPRegressor()
                    grid_ann = GridSearchCV(ann, param_grid=grid_values_ann, scoring='neg_mean_squared_error', n_jobs=5)
                    grid_ann.fit(X_train_norm, y_train.to_numpy().reshape(-1))
                    end_ann = time.time()
                    print('Time elapsed ANN: ', int((end_ann - start_ann) / 60))
                    with open(path_to_save + '/ANN_'+key_type+'.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_ann, file)
                    with open(path_to_save + '/Computation_time_ANN_4.txt', 'a+') as f: #save the computation time
                        f.write('Computation time: ' + str(int((end_ann - start_ann) / 60)) + ' minutes')
                elif model_state[algo] == 'saved':
                    print('---------------ANN LOADING---------------')
                    with open(path_to_save + '/ANN_'+key_type+'.pkl', 'rb') as file: #load the trained model
                        grid_ann = pickle.load(file)
                y_pred_ann_train = grid_ann.predict(X_train_norm)
                y_pred_ann_test = grid_ann.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_ann_test[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(y_test_not_zero - y_pred_not_zero), y_test_not_zero)).mean()
                mae = mean_absolute_error(y_test, y_pred_ann_test) * 3600
                rmae = mae / mean_WT_test
                rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_ann_train)) * 3600
                rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_ann_test)) * 3600
                rrmse_train = rmse_train / mean_WT_train
                rrmse_test = rmse_test / mean_WT_test
                bias = np.mean([y_test.to_numpy()[i] - y_pred_ann_test[i] for i in range(len(y_test))]) * 3600
                results_dict['ANN'] = [grid_ann.best_estimator_, round(rmse_train, 2), round(rmse_test, 2), round(rrmse_train, 2),
                                      round(rrmse_test, 2), round(rmae, 2), round(mae, 2), round(re, 2), round(bias, 2)]
            elif algo == 'LR':
                print('---------------LR FITTING---------------')
                reg = LinearRegression()
                reg.fit(X_train_norm, y_train)
                y_pred_reg_train = reg.predict(X_train_norm)
                y_pred_reg_test = reg.predict(X_test_norm)
                y_pred_not_zero = np.array(y_pred_reg_test[y_test.loc[y_test.Real_WT > (2 / 3600)].index])
                re = (np.divide(np.absolute(np.array(y_test_not_zero).reshape(-1) - np.array(y_pred_not_zero).reshape(-1)), np.array(y_test_not_zero).reshape(-1))).mean()
                mae = mean_absolute_error(y_test, y_pred_reg_test) * 3600
                rmae = mae / mean_WT_test
                rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_reg_train)) * 3600
                rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_reg_test)) * 3600
                rrmse_train = rmse_train / mean_WT_train
                rrmse_test = rmse_test / mean_WT_test
                bias = np.mean([y_test.to_numpy()[i] - y_pred_reg_test[i] for i in range(len(y_test))]) * 3600
                results_dict['LR'] = [reg.coef_, round(rmse_train, 2), round(rmse_test, 2), round(rrmse_train, 2),
                                      round(rrmse_test, 2), round(rmae, 2), round(mae, 2), round(re, 2), round(bias, 2)]

            elif algo == 'QTD':
                print('---------------QTD Prediction---------------')
                re = (np.divide(np.absolute(y_test_not_zero - y_QTD_test_not_zero), y_test_not_zero)).mean()
                mae = mean_absolute_error(y_test, y_QTD_test) * 3600
                rmae = mean_absolute_error(y_test, y_QTD_test) * 3600/ mean_WT_test
                rmse_test = math.sqrt(mean_squared_error(y_test, y_QTD_test)) * 3600
                rrmse_test = rmse_test / mean_WT_test
                bias = np.mean([y_test.to_numpy()[i] - y_QTD_test.to_numpy()[i] for i in range(len(y_test))]) * 3600
                results_dict['QTD'] = [None, 0, round(rmse_test, 2), 0,
                                      round(rrmse_test, 2), round(rmae, 2), round(mae, 2), round(re, 2), round(bias, 2)]

            elif algo == 'LES':
                print('---------------LES Prediction---------------')
                re = (np.divide(np.absolute(y_test_not_zero - y_LES_test_not_zero), y_test_not_zero)).mean()
                mae = mean_absolute_error(y_test, y_LES_test) * 3600
                rmae = mae / mean_WT_test
                rmse_test = math.sqrt(mean_squared_error(y_test, y_LES_test)) * 3600
                rrmse_test = rmse_test / mean_WT_test
                bias = np.mean([y_test.to_numpy()[i] - y_LES_test.to_numpy()[i] for i in range(len(y_test))]) * 3600
                results_dict['LES'] = [None, 0, round(rmse_test, 2), 0,
                                      round(rrmse_test, 2), round(rmae, 2), round(mae, 2), round(re, 2), round(bias, 2)]
            elif algo == 'HOL':
                print('---------------HOL Prediction---------------')
                re = (np.divide(np.absolute(y_test_not_zero - y_HOL_test_not_zero), y_test_not_zero)).mean()
                mae = mean_absolute_error(y_test, y_HOL_test) * 3600
                rmae = mae / mean_WT_test
                rmse_test = math.sqrt(mean_squared_error(y_test, y_HOL_test)) * 3600
                rrmse_test = rmse_test / mean_WT_test
                bias = np.mean([y_test.to_numpy()[i] - y_HOL_test.to_numpy()[i] for i in range(len(y_test))]) * 3600
                results_dict['HOL'] = [None, 0, round(rmse_test, 2), 0,
                                      round(rrmse_test, 2), round(rmae, 2), round(mae, 2), round(re, 2), round(bias, 2)]


        with open(path_to_save+ '/results_dict_got_service'+ key_type+'.p', 'wb') as fp:
            pickle.dump(results_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        rmse_train_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        rmse_test_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        rrmse_train_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        rrmse_test_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        rmae_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        mae_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        re_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        bias_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        hyper_parameters = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
        labels = []

        df_results = pd.DataFrame(columns=['RMSE_train', 'RMSE_test', 'RRMSE_train', 'RRMSE_test', 'RMAE', 'MAE', 'RE', 'Bias', 'HyperParameters'],
                                  index=['LR', 'RF', 'SVR', 'GBR', 'ANN', 'QTD', 'LES', 'HOL'])

        for algo, results in results_dict.items():
            hyper_parameters[algo].append(results[0])
            rmse_train_metrics[algo].append(results[1])
            rmse_test_metrics[algo].append(results[2])
            rrmse_train_metrics[algo].append(results[3])
            rrmse_test_metrics[algo].append(results[4])
            rmae_metrics[algo].append(results[5])
            mae_metrics[algo].append(results[6])
            re_metrics[algo].append(results[7])
            bias_metrics[algo].append(results[8])

        df_results['RMSE_train'] = np.concatenate(list(rmse_train_metrics.values()))
        df_results['RMSE_test'] = np.concatenate(list(rmse_test_metrics.values()))
        df_results['RRMSE_train'] = np.concatenate(list(rrmse_train_metrics.values()))
        df_results['RRMSE_test'] = np.concatenate(list(rrmse_test_metrics.values()))
        df_results['MAE'] = np.concatenate(list(mae_metrics.values()))
        df_results['RMAE'] = np.concatenate(list(rmae_metrics.values()))
        df_results['RE'] = np.concatenate(list(re_metrics.values()))
        df_results['Bias'] = np.concatenate(list(bias_metrics.values()))
        df_results['HyperParameters'] = list(hyper_parameters.values())

        df_results.to_csv(path_to_save+'/Results_got_service_'+key_type+'.csv')

    end = time.time()
    print('Time elapsed: ', int((end-start)/60))

