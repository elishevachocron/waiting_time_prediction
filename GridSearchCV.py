import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle
from const import *
from Data_loading import *
from Metrics_comp import *


if __name__ == '__main__':

    for experiment in [0]:
        print('--------------------------------------Experiment ' + str(experiment)+'--------------------------------------')
        path_to_load = path + str(experiment)
        path_to_save = path + str(experiment) + '\\Final_results\\Arik_features'

        results_dict = {'RF': {}, 'ANN': {}, 'LR': {}, 'QTP': {}, 'QTP_weighted': {}, 'LES': {}, 'HOL': {}}

        data = pd.read_csv(path_to_load+'/Arik_features_imple.csv')
        data_class = DataLoader(num=experiment, data=data, week_split=4, abandonment_removed_train=1, abandonment_removed_test=1,  features_list=features, beginning=exp_beginning[experiment])
        data_class.print_summary()

        #Data loading
        X_train, X_train_checked, y_train, mean_WT_train = data_class.extract_train()
        X_test, X_test_checked, y_test, mean_WT_test = data_class.extract_test()
        y_QTP_test, y_QTP_weighted_test, y_QTP_weighted_bis_test = data_class.extract_QT_predictors_test()
        y_HOL_test, y_LES_test = data_class.extract_snapshot_test()
        #y_test_not_zero, y_QTP_test_not_zero, y_QTP_weighted_test_not_zero, y_LES_test_not_zero, y_HOL_test_not_zero = data_class.extract_no_zero()

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
                    grid_rf.fit(X_train_norm, y_train)
                    end_rf = time.time()
                    print('Time elapsed RF: ', int((end_rf - start_rf) / 60))
                    with open(path_to_save + '/RF.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_rf, file)
                    with open(path_to_save + '/Computation_time_RF.txt', 'a+') as f: #save the computation time
                        f.write(' Computation time: ' + str(int((end_rf - start_rf) / 60)) + ' minutes')
                elif model_state[algo] == 'saved':
                    print('---------------RF LOADING---------------')
                    with open(path_to_save + '/RF.pkl', 'rb') as file: #load the trained model
                        grid_rf = pickle.load(file)

                y_pred_rf_train = grid_rf.predict(X_train_norm)
                y_pred_rf_test = grid_rf.predict(X_test_norm)
                evaluation_train = Evaluation(y=y_train, y_hat=y_pred_rf_train, quantile=quantile)
                evaluation_test = Evaluation(y=y_test, y_hat=y_pred_rf_test, quantile=quantile)
                re_train, mae_train, rmae_train, rmse_train, rrmse_train, bias_train = evaluation_train.metrics_computation()
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)

                results_dict['RF'] = [re_train, re_test, mae_train, mae_test, rmae_train, rmae_test, rmse_train,
                                      rmse_test, rrmse_train, rrmse_test, bias_train, bias_test, mean_loss, rmean_loss, grid_rf.best_estimator_]

            elif algo == 'SVR':
                if model_state[algo] == 'fit':
                    start_svr = time.time()
                    print('---------------SVR FITTING---------------')
                    svr = SVR()
                    grid_svr = GridSearchCV(svr, param_grid=grid_values_svr, scoring='neg_mean_squared_error', n_jobs=5)
                    grid_svr.fit(X_train_norm, y_train)
                    end_svr = time.time()
                    print('Time elapsed SVR: ', int((end_svr - start_svr) / 60))
                    with open(path_to_save + '/SVR.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_svr, file)
                    with open(path_to_save + '/Computation_time_SVR.txt', 'a+') as f: #save the computation time
                        f.write('Computation time: ' + str(int((end_svr - start_svr) / 60)) + ' minutes')
                elif model_state[algo] == 'saved':
                    print('---------------SVR LOADING---------------')
                    with open(path_to_save + '/SVR.pkl', 'rb') as file: #load the trained model
                        grid_svr = pickle.load(file)
                y_pred_svr_train = grid_svr.predict(X_train_norm)
                y_pred_svr_test = grid_svr.predict(X_test_norm)
                evaluation_train = Evaluation(y=y_train, y_hat=y_pred_svr_train, quantile=quantile)
                evaluation_test = Evaluation(y=y_test, y_hat=y_pred_svr_test, quantile=quantile)
                re_train, mae_train, rmae_train, rmse_train, rrmse_train, bias_train = evaluation_train.metrics_computation()
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)

                results_dict['SVR'] = [re_train, re_test, mae_train, mae_test, rmae_train, rmae_test, rmse_train,
                                      rmse_test, rrmse_train, rrmse_test, bias_train, bias_test, mean_loss, rmean_loss, grid_svr.best_estimator_]
            elif algo == 'GBR':
                if model_state[algo] == 'fit':
                    start_gbr = time.time()
                    print('---------------GBR FITTING---------------')
                    gbr = GradientBoostingRegressor()
                    grid_gbr = GridSearchCV(gbr, param_grid=grid_values_gbr, scoring='neg_mean_squared_error', n_jobs=8)
                    grid_gbr.fit(X_train_norm, y_train)
                    end_gbr = time.time()
                    print('Time elapsed GBR: ', int((end_gbr - start_gbr) / 60))
                    with open(path_to_save + '/GBR.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_gbr, file)
                    with open(path_to_save + '/Computation_time_GBR.txt', 'a+') as f:  # save the computation time
                        f.write('Computation time: ' + str(int((end_gbr - start_gbr) / 60)) + ' minutes')
                elif model_state[algo] == 'saved':
                    print('---------------GBR LOADING---------------')
                    with open(path_to_save + '/GBR.pkl', 'rb') as file: #load the trained model
                        grid_gbr = pickle.load(file)
                y_pred_gbr_train = grid_gbr.predict(X_train_norm)
                y_pred_gbr_test = grid_gbr.predict(X_test_norm)
                evaluation_train = Evaluation(y=y_train, y_hat=y_pred_gbr_train, quantile=quantile)
                evaluation_test = Evaluation(y=y_test, y_hat=y_pred_gbr_test, quantile=quantile)
                re_train, mae_train, rmae_train, rmse_train, rrmse_train, bias_train = evaluation_train.metrics_computation()
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)

                results_dict['GBR'] = [re_train, re_test, mae_train, mae_test, rmae_train, rmae_test, rmse_train,
                                      rmse_test, rrmse_train, rrmse_test, bias_train, bias_test, mean_loss, rmean_loss, grid_gbr.best_estimator_]


            elif algo == 'ANN':
                if model_state[algo] == 'fit':
                    start_ann = time.time()
                    print('---------------ANN FITTING---------------')
                    ann = MLPRegressor()
                    ann.out_activation_ = 'relu'
                    grid_ann = GridSearchCV(ann, param_grid=grid_values_ann, scoring='neg_mean_squared_error', n_jobs=5)
                    grid_ann.fit(X_train_norm, y_train)
                    end_ann = time.time()
                    print('Time elapsed ANN: ', int((end_ann - start_ann) / 60))
                    with open(path_to_save + '/ANN.pkl', 'wb') as file:  # save the trained model
                        pickle.dump(grid_ann, file)
                    with open(path_to_save + '/Computation_time_ANN.txt', 'a+') as f: #save the computation time
                        f.write('Computation time: ' + str(int((end_ann - start_ann) / 60)) + ' minutes')
                elif model_state[algo] == 'saved':
                    print('---------------ANN LOADING---------------')
                    with open(path_to_save + '/ANN.pkl', 'rb') as file: #load the trained model
                        grid_ann = pickle.load(file)
                y_pred_ann_train = grid_ann.predict(X_train_norm)
                y_pred_ann_test = grid_ann.predict(X_test_norm)
                evaluation_train = Evaluation(y=y_train, y_hat=y_pred_ann_train, quantile=quantile)
                evaluation_test = Evaluation(y=y_test, y_hat=y_pred_ann_test, quantile=quantile)
                re_train, mae_train, rmae_train, rmse_train, rrmse_train, bias_train = evaluation_train.metrics_computation()
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)

                results_dict['ANN'] = [re_train, re_test, mae_train, mae_test, rmae_train, rmae_test, rmse_train,
                                      rmse_test, rrmse_train, rrmse_test, bias_train, bias_test, mean_loss, rmean_loss, grid_ann.best_estimator_]
            elif algo == 'LR':
                print('---------------LR FITTING---------------')
                reg = LinearRegression()
                reg.fit(X_train_norm, y_train)
                y_pred_reg_train = reg.predict(X_train_norm)
                y_pred_reg_test = reg.predict(X_test_norm)
                evaluation_train = Evaluation(y=y_train, y_hat=y_pred_reg_train, quantile=quantile)
                evaluation_test = Evaluation(y=y_test, y_hat=y_pred_reg_test, quantile=quantile)
                re_train, mae_train, rmae_train, rmse_train, rrmse_train, bias_train = evaluation_train.metrics_computation()
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)

                results_dict['LR'] = [re_train, re_test, mae_train, mae_test, rmae_train, rmae_test, rmse_train,
                                      rmse_test, rrmse_train, rrmse_test, bias_train, bias_test, mean_loss, rmean_loss, reg.coef_]

            elif algo == 'QTP':
                print('---------------QTP Prediction---------------')
                evaluation_test = Evaluation(y=y_test, y_hat=y_QTP_test, quantile=quantile)
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)
                results_dict['QTP'] = ['-', re_test, '-', mae_test, '-',
                                       rmae_test, '-', rmse_test, '-', rrmse_test, '-', bias_test, mean_loss, rmean_loss, '-']

            elif algo == 'QTP_weighted':
                print('---------------QTP weighted Prediction---------------')
                evaluation_test = Evaluation(y=y_test, y_hat=y_QTP_weighted_test, quantile=quantile)
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)
                results_dict['QTP_weighted'] = ['-', re_test, '-', mae_test, '-',
                                       rmae_test, '-', rmse_test, '-', rrmse_test, '-', bias_test, mean_loss, rmean_loss, '-']

            elif algo == 'QTP_weighted_bis':
                print('---------------QTP weighted bis Prediction---------------')
                evaluation_test = Evaluation(y=y_test, y_hat=y_QTP_weighted_bis_test, quantile=quantile)
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)
                results_dict['QTP_weighted_bis'] = ['-', re_test, '-', mae_test, '-',
                                       rmae_test, '-', rmse_test, '-', rrmse_test, '-', bias_test, mean_loss, rmean_loss, '-']

            elif algo == 'LES':
                print('---------------LES Prediction---------------')
                evaluation_test = Evaluation(y=y_test, y_hat=y_LES_test, quantile=quantile)
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)
                results_dict['LES'] = ['-', re_test, '-', mae_test, '-',
                                       rmae_test, '-', rmse_test, '-', rrmse_test, '-', bias_test, mean_loss, rmean_loss, '-']

            elif algo == 'HOL':
                print('---------------HOL Prediction---------------')
                evaluation_test = Evaluation(y=y_test, y_hat=y_HOL_test, quantile=quantile)
                re_test, mae_test, rmae_test, rmse_test, rrmse_test, bias_test = evaluation_test.metrics_computation()
                mean_loss, rmean_loss = evaluation_test.MeanLoss(delta=delta)
                results_dict['HOL'] = ['-', re_test, '-', mae_test, '-',
                                       rmae_test, '-', rmse_test, '-', rrmse_test, '-', bias_test, mean_loss, rmean_loss, '-']


        with open(path_to_save+ '/results_dict_got_service.p', 'wb') as fp:
            pickle.dump(results_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        rmse_train_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}
        rmse_test_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}

        rrmse_train_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}
        rrmse_test_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}

        rmae_train_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}
        rmae_test_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [],'HOL': []}

        mae_train_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}
        mae_test_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}

        re_train_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}
        re_test_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}

        bias_train_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}
        bias_test_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}

        mean_loss_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}
        rmean_loss_metrics = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}

        hyper_parameters = {'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTP': [], 'QTP_weighted': [], 'QTP_weighted_bis': [], 'LES': [], 'HOL': []}
        labels = []

        df_results = pd.DataFrame(columns=['RMSE_train', 'RMSE_test', 'RRMSE_train', 'RRMSE_test', 'RMAE_train', 'RMAE_test',
                                           'MAE_train', 'MAE_test', 'RE_train', 'RE_test', 'Bias_train', 'Bias_test', 'MeanLoss_'+str(delta), 'RMeanLoss_'+str(delta), 'HyperParameters'],
                                  index=['LR', 'RF', 'SVR', 'GBR', 'ANN', 'QTP', 'QTP_weighted', 'QTP_weighted_bis', 'LES', 'HOL'])

        for algo, results in results_dict.items():
            re_train_metrics[algo].append(results[0])
            re_test_metrics[algo].append(results[1])
            mae_train_metrics[algo].append(results[2])
            mae_test_metrics[algo].append(results[3])
            rmae_train_metrics[algo].append(results[4])
            rmae_test_metrics[algo].append(results[5])
            rmse_train_metrics[algo].append(results[6])
            rmse_test_metrics[algo].append(results[7])
            rrmse_train_metrics[algo].append(results[8])
            rrmse_test_metrics[algo].append(results[9])
            bias_train_metrics[algo].append(results[10])
            bias_test_metrics[algo].append(results[11])
            mean_loss_metrics[algo].append(results[12])
            rmean_loss_metrics[algo].append(results[13])
            hyper_parameters[algo].append(results[14])


        df_results['RMSE_train'] = np.concatenate(list(rmse_train_metrics.values()))
        df_results['RMSE_test'] = np.concatenate(list(rmse_test_metrics.values()))

        df_results['RRMSE_train'] = np.concatenate(list(rrmse_train_metrics.values()))
        df_results['RRMSE_test'] = np.concatenate(list(rrmse_test_metrics.values()))

        df_results['MAE_train'] = np.concatenate(list(mae_train_metrics.values()))
        df_results['MAE_test'] = np.concatenate(list(mae_test_metrics.values()))

        df_results['RMAE_train'] = np.concatenate(list(rmae_train_metrics.values()))
        df_results['RMAE_test'] = np.concatenate(list(rmae_test_metrics.values()))

        df_results['RE_train'] = np.concatenate(list(re_train_metrics.values()))
        df_results['RE_test'] = np.concatenate(list(re_test_metrics.values()))

        df_results['Bias_train'] = np.concatenate(list(bias_train_metrics.values()))
        df_results['Bias_test'] = np.concatenate(list(bias_test_metrics.values()))

        df_results['MeanLoss_'+str(delta)] = np.concatenate(list(mean_loss_metrics.values()))
        df_results['RMeanLoss_'+str(delta)] = np.concatenate(list(rmean_loss_metrics.values()))

        df_results['HyperParameters'] = list(hyper_parameters.values())

        df_results.to_csv(path_to_save + '/Results_quantile_' + str(quantile) + '.csv')

    end = time.time()
    print('Time elapsed: ', int((end-start)/60))

