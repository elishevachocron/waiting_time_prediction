import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from const import *
from Data_loading import *
import pickle
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from Data_loading import *

np.random.seed(42)

def Plot_error(y_checked, rf_checked, svr_checked, gbr_checked, ann_checked, lr_checked, QTP_checked, QTP_weighted_checked, LES_checked, HOL_checked,set, title, exp):
    plt.figure()
    plt.scatter(y_checked, rf_checked, label='RF')
    plt.scatter(y_checked, svr_checked, label='SVR')
    plt.scatter(y_checked, gbr_checked, label='GBR')
    plt.scatter(y_checked, ann_checked, label='ANN')
    plt.scatter(y_checked, lr_checked, label='LR')
    plt.scatter(y_checked, QTP_checked, label='QTP')
    plt.scatter(y_checked, QTP_weighted_checked, label='QTP_weighted')
    plt.scatter(y_checked, LES_checked, label='LES')
    plt.scatter(y_checked, HOL_checked, label='HOL')
    plt.plot([0, np.max(y_checked)], [0, np.max(y_checked)])
    plt.xlabel('Real Waiting time')
    plt.ylabel('Waiting time prediction')
    plt.legend()
    plt.savefig(path_error_analysis + '/' + set + '.png')

def Outliers_Analysis(X_checked, y_checked, y_pred, algo, title, exp):
    #diff = np.divide(np.abs(y_checked - y_pred), y_checked)
    diff = np.abs(y_checked - y_pred)
    idx = diff.argsort()[-10:]
    df = pd.concat([X_checked.iloc[idx], pd.Series(y_checked[idx], name='Real_WT', index=X_checked.iloc[idx].index),
                       pd.Series(y_pred[idx], name=algo+'_prediction', index=X_checked.iloc[idx].index)], axis=1)
    df.to_csv(path_error_analysis + '/' + algo +'_'+title+'_outliers.csv')

if __name__ == '__main__':

    for exp in [0, 12]:
        print('--------------------------------------Experiment ' + str(exp)+'--------------------------------------')
        path_to_load = path + str(exp)
        path_to_save = path + str(exp) + '\\Final_results'
        path_error_analysis = path_to_save + '\\Error_analysis'
        data = pd.read_csv(path_to_load+'/Simulation_corrected_full_imple.csv')

        data_class = DataLoader(num=exp, data=data, week_split=4, abandonment_removed_train=1, abandonment_removed_test=1,
                                features_list=features, features_to_check=features_to_check, beginning=exp_beginning[exp])
        data_class.print_summary()

        #Data loading
        X_train, X_train_checked, y_train, mean_WT_train = data_class.extract_train()
        X_test, X_test_checked, y_test, mean_WT_test = data_class.extract_test()

        y_QTP_train, y_QTP_weighted_train = data_class.extract_QT_predictors_train()
        y_QTP_test, y_QTP_weighted_test, y_QTP_weighted_bis_test  = data_class.extract_QT_predictors_test()

        y_HOL_train, y_LES_train = data_class.extract_snapshot_train()
        y_HOL_test, y_LES_test = data_class.extract_snapshot_test()

        #y_test_not_zero, y_QTP_test_not_zero, y_QTP_weighted_test_not_zero, y_LES_test_not_zero, y_HOL_test_not_zero = data_class.extract_no_zero()

        # Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

        #Train and Test set Histogram

        max_value = max(int(np.max(y_train)), int(np.max(y_test)))
        min_value = min(int(np.min(y_train)), int(np.min(y_test)))
        bins = np.linspace(min_value, max_value, int(max_value/50))

        plt.figure()
        plt.hist(y_train, bins, weights=[1/len(y_train)]*len(y_train), alpha=0.5, label='Train set')
        plt.hist(y_test, bins, weights=[1/len(y_test)]*len(y_test), alpha=0.5, label='Test set')
        plt.xlabel('Waiting time (sec)')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig(path_error_analysis+'/Histogram.png')

        # Train and Test statistics values and Boxplot
        s_train = pd.Series(y_train, name='Train')
        s_test = pd.Series(y_test, name='Test')

        print('Statistics train: \n', s_train. describe())
        print('90%: \n', s_train.quantile(0.9))
        print('95%: \n', s_train.quantile(0.95))
        print('Statistics test: \n', s_test.describe())
        print('90%: \n', s_test.quantile(0.9))
        print('95%: \n', s_test.quantile(0.95))

        df = pd.concat([s_train, s_test], axis=1).reset_index()
        plt.figure()
        boxplot = df.boxplot(column=['Train', 'Test'])
        plt.savefig(path_error_analysis+'/Boxplot.png')

        for algo in algorithms_used:
            if algo == 'RF':
                print('---------------RF LOADING---------------')
                with open(path_to_save + '/RF.pkl', 'rb') as file:  # load the trained model
                    grid_rf = pickle.load(file)
                #y_pred_rf_train = grid_rf.best_estimator_.oob_prediction_
                y_pred_rf_train = grid_rf.predict(X_train_norm)
                y_pred_rf_test = grid_rf.predict(X_test_norm)

                #Features importance
                plt.figure()
                plt.bar(np.arange(len(features)), grid_rf.best_estimator_.feature_importances_)
                plt.xticks(np.arange(len(features)), np.arange(0, len(features)))
                plt.savefig(path_error_analysis + '/' + algo + '.png')

            elif algo == 'SVR':
                print('---------------SVR LOADING---------------')
                with open(path_to_save + '/SVR.pkl', 'rb') as file:  # load the trained model
                    grid_svr = pickle.load(file)
                y_pred_svr_train = grid_svr.predict(X_train_norm)
                y_pred_svr_test = grid_svr.predict(X_test_norm)

            elif algo == 'GBR':
                print('---------------GBR LOADING---------------')
                with open(path_to_save + '/GBR.pkl', 'rb') as file:  # load the trained model
                    grid_gbr = pickle.load(file)
                y_pred_gbr_train = grid_gbr.predict(X_train_norm)
                y_pred_gbr_test = grid_gbr.predict(X_test_norm)
                #Features importance
                plt.figure()
                plt.bar(np.arange(len(features)), grid_gbr.best_estimator_.feature_importances_)
                plt.xticks(np.arange(len(features)), np.arange(0, len(features)))
                plt.savefig(path_error_analysis + '/' + algo + '.png')

            elif algo == 'ANN':
                print('---------------ANN LOADING---------------')
                with open(path_to_save + '/ANN.pkl', 'rb') as file:  # load the trained model
                    grid_ann = pickle.load(file)
                y_pred_ann_train = grid_ann.predict(X_train_norm)
                y_pred_ann_test = grid_ann.predict(X_test_norm)


            elif algo == 'LR':
                print('---------------LR FITTING---------------')
                reg = LinearRegression()
                reg.fit(X_train_norm, y_train)
                y_pred_reg_train = np.array(reg.predict(X_train_norm)).reshape(-1)
                y_pred_reg_test = np.array(reg.predict(X_test_norm)).reshape(-1)

        not_zero_idx_train = np.where(y_train != 0)[0]
        not_zero_idx_test = np.where(y_test != 0)[0]

        idx_train = random.sample(list(not_zero_idx_train), int(len(y_train) * 0.01))
        idx_test = random.sample(list(not_zero_idx_test), int(len(y_test) * 0.05))

        rf_checked_train = y_pred_rf_train[idx_train]
        svr_checked_train = y_pred_svr_train[idx_train]
        gbr_checked_train = y_pred_gbr_train[idx_train]
        ann_checked_train = y_pred_ann_train[idx_train]
        lr_checked_train = y_pred_reg_train[idx_train]
        QTP_checked_train = y_QTP_train[idx_train]
        QTP_weigthed_checked_train = y_QTP_weighted_train[idx_train]
        LES_checked_train = y_LES_train[idx_train]
        HOL_checked_train = y_HOL_train[idx_train]
        y_checked_train = y_train[idx_train]

        rf_checked_test = y_pred_rf_test[idx_test]
        svr_checked_test = y_pred_svr_test[idx_test]
        gbr_checked_test = y_pred_gbr_test[idx_test]
        ann_checked_test = y_pred_ann_test[idx_test]
        lr_checked_test = y_pred_reg_test[idx_test]
        QTP_checked_test = y_QTP_test[idx_test]
        QTP_weigthed_checked_test = y_QTP_weighted_test[idx_test]
        LES_checked_test = y_LES_test[idx_test]
        HOL_checked_test = y_HOL_test[idx_test]
        y_checked_test = y_test[idx_test]
        X_test_checked['QTP_weighted'] *= 3600
        X_test_checked['QTP_weighted_bis'] *= 3600
        X_test_checked['QTP_data_based'] *= 3600
        X_checked = X_test_checked.iloc[idx_test]

        print('hello')

        #Figure Error Analysis
        Plot_error(y_checked_train, rf_checked_train, svr_checked_train, gbr_checked_train, ann_checked_train, lr_checked_train,
                   QTP_checked_train, QTP_weigthed_checked_train, LES_checked_train, HOL_checked_train, 'Train', title, exp)
        Plot_error(y_checked_test, rf_checked_test, svr_checked_test, gbr_checked_test, ann_checked_test, lr_checked_test,
                   QTP_checked_test, QTP_weigthed_checked_test, LES_checked_test, HOL_checked_test, 'Test', title, exp)


        #Outliers Analysis

        Outliers_Analysis(X_checked, y_checked_test, rf_checked_test, 'RF', title, exp)
        Outliers_Analysis(X_checked, y_checked_test, svr_checked_test, 'SVR', title, exp)
        Outliers_Analysis(X_checked, y_checked_test, gbr_checked_test, 'GBR', title, exp)
        Outliers_Analysis(X_checked, y_checked_test, ann_checked_test, 'ANN', title, exp)
        Outliers_Analysis(X_checked, y_checked_test, lr_checked_test, 'LR', title, exp)
        Outliers_Analysis(X_checked, y_checked_test, QTP_checked_test, 'QTP', title, exp)
        Outliers_Analysis(X_checked, y_checked_test, QTP_weigthed_checked_test, 'QTP_weighted', title, exp)
        Outliers_Analysis(X_checked, y_checked_test, LES_checked_test, 'LES', title, exp)
        Outliers_Analysis(X_checked, y_checked_test, HOL_checked_test, 'HOL', title, exp)