import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from const import *
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from Data_loading import *
from Metrics_comp import *
np.random.seed(42)

sns.set()

# labels = ['0', '5', '10', '15']
# #dict_exp = {1: '1.0', 2: '2.0', 3: '2.1', 4: '2.2', 5: '2.3', 6: '2.4', 7: '2.5', 8: '2.6', 9: '3.0', 10: '3.1', 11: '3.2', 0: '4'}
# dict_exp = {0: '4', 12: '4.1'}
# #for exp in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]:
# for exp in dict_exp.keys():
#     print('Exp: ', exp)
#     MeanLoss_LR = []
#     MeanLoss_RF = []
#     MeanLoss_SVR = []
#     MeanLoss_GBR = []
#     MeanLoss_ANN = []
#     MeanLoss_QTP = []
#     MeanLoss_HOL = []
#     MeanLoss_LES = []
#
#     for threshold in [0, 5, 10, 15]:
#         print('Threshold: ', threshold)
#         df = pd.read_csv(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results/MeanLoss('+str(threshold)+').csv')
#         MeanLoss_LR.append(float(df.loc[df['Unnamed: 0'] == 'LR'][str(exp)]))
#         MeanLoss_RF.append(float(df.loc[df['Unnamed: 0'] == 'RF'][str(exp)]))
#         MeanLoss_SVR.append(float(df.loc[df['Unnamed: 0'] == 'SVR'][str(exp)]))
#         MeanLoss_GBR.append(float(df.loc[df['Unnamed: 0'] == 'GBR'][str(exp)]))
#         MeanLoss_ANN.append(float(df.loc[df['Unnamed: 0'] == 'ANN'][str(exp)]))
#         MeanLoss_QTP.append(float(df.loc[df['Unnamed: 0'] == 'QTD'][str(exp)]))
#         MeanLoss_HOL.append(float(df.loc[df['Unnamed: 0'] == 'HOL'][str(exp)]))
#         MeanLoss_LES.append(float(df.loc[df['Unnamed: 0'] == 'LES'][str(exp)]))
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.10 # the width of the bars
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - 3.5*width, MeanLoss_LR, width, label='LR')
#     rects2 = ax.bar(x - 2.5*width, MeanLoss_RF, width, label='RF')
#     rects3 = ax.bar(x - 1.5*width, MeanLoss_SVR, width, label='SVR')
#     rects4 = ax.bar(x - 0.5*width, MeanLoss_GBR, width, label='GBR')
#     rects5 = ax.bar(x + 0.5*width, MeanLoss_ANN, width, label='ANN')
#     rects6 = ax.bar(x + 1.5*width, MeanLoss_QTP, width, label='QTP')
#     rects7 = ax.bar(x + 2.5*width, MeanLoss_HOL, width, label='HOL')
#     rects8 = ax.bar(x + 3.5*width, MeanLoss_LES, width, label='LES')
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Mean Loss (sec)')
#     ax.set_xlabel('Delta (sec)')
#     ax.set_title('MeanLoss experiment ' + dict_exp[exp] )
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()
#
#     fig.tight_layout()
#     plt.savefig(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results\MeanLoss/Mean_loss_exp_'+dict_exp[exp]+'.png')
#
# labels = ['1.0', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '3.0', '3.1', '3.2', '4']
# df = pd.read_csv(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results/RMeanLoss(10).csv')
# LR = np.append(df.iloc[0, 2:].to_numpy(), float(df.iloc[0, 1]))
# RF = np.append(df.iloc[1, 2:].to_numpy(), float(df.iloc[1, 1]))
# ANN = np.append(df.iloc[4, 2:].to_numpy(), float(df.iloc[4, 1]))
# QTP = np.append(df.iloc[5, 2:].to_numpy(), float(df.iloc[5, 1]))
#
# x = np.arange(len(labels))  # the label locations
# width = 0.20 # the width of the bars
#
# fig, ax = plt.subplots()
# rects3 = ax.bar(x - 1.5*width, LR, width, label='LR')
# rects4 = ax.bar(x - 0.5*width, RF, width, label='RF')
# rects5 = ax.bar(x + 0.5*width, ANN, width, label='ANN')
# rects6 = ax.bar(x + 1.5*width, QTP, width, label='QTP')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('RMeanLoss (delta = 10 sec)')
# ax.set_xlabel('Experiment')
# ax.set_title('RMeanLoss')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
# fig.tight_layout()
# plt.savefig(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results\MeanLoss/RMeanLoss.png')
#
#

def loss_function (y_hat, y, threshold):
    difference_abs = np.absolute(y_hat - y) - np.full(len(y), threshold)
    difference_abs[difference_abs < 0] = 0
    return np.mean(difference_abs)

threshold = int(input('Threshold Definition: '))
Results_MeanLoss = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=algorithms_used)
Results_RMeanLoss = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=algorithms_used)

# for exp in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 12]:
for exp in [0]:
    print('--------------------------------------Experiment ' + str(exp) + '--------------------------------------')
    path_to_load = path + str(exp)
    path_to_save = path + str(exp) + '\\Final_results'
    path_error_analysis = path_to_save + '\\Error_analysis'
    data = pd.read_csv(path_to_load + '/Simulation_corrected_full_imple.csv')

    data_class = DataLoader(num=exp, data=data, week_split=4, abandonment_removed_train=1, abandonment_removed_test=1,
                            features_list=features, features_to_check=features_to_check, beginning=exp_beginning[exp])
    data_class.print_summary()

    # Data loading
    X_train, X_train_checked, y_train, mean_WT_train = data_class.extract_train()
    X_test, X_test_checked, y_test, mean_WT_test = data_class.extract_test()

    y_QTP_train, y_QTP_weighted_train = data_class.extract_QT_predictors_train()
    y_QTP_test, y_QTP_weighted_test = data_class.extract_QT_predictors_test()

    y_HOL_train, y_LES_train = data_class.extract_snapshot_train()
    y_HOL_test, y_LES_test = data_class.extract_snapshot_test()

    # Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    for algo in algorithms_used:
        if algo == 'RF':
            print('---------------RF LOADING---------------')
            with open(path_to_save + '/RF.pkl', 'rb') as file:  # load the trained model
                grid_rf = pickle.load(file)
            y_pred_rf_train = grid_rf.predict(X_train_norm)
            y_pred_rf_test = grid_rf.predict(X_test_norm)

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
            y_pred_reg_train = reg.predict(X_train_norm)
            y_pred_reg_test = reg.predict(X_test_norm)

    predictions = {'LR': y_pred_reg_test, 'RF': y_pred_rf_test, 'SVR': y_pred_svr_test, 'GBR': y_pred_gbr_test,
                   'ANN': y_pred_ann_test, 'QTP': y_QTP_test, 'QTP_weighted': y_QTP_weighted_test,
                   'LES': y_LES_test, 'HOL': y_HOL_test}


    for algo in algorithms_used:
        if algo == 'QTP':
            print('hello')
        Results_MeanLoss[exp][algo] = round(loss_function(predictions[algo], y_test, threshold), 2)

    Results_RMeanLoss[exp] = Results_MeanLoss[exp]/y_test.mean()

    print('hello')

Results_MeanLoss.to_csv(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results/Corrected_MeanLoss('+str(threshold)+').csv')
Results_RMeanLoss.to_csv(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results/Corrected_RMeanLoss('+str(threshold)+').csv')

#plot the results

#labels = ['1.0', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '3.0', '3.1', '3.2', '4']
labels = ['0', '5', '10', '15']
#dict_exp = {1: '2.0', 2: '2.0', 3: '2.0', 4: '2.0', 5:'2.0', 6: '2.0', 7: '2.0', 8: '2.0', 9: '2.0', 10: '2.0', 11: '2.0', 0: '4.0'}

#for exp in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]:
for exp in dict_exp.keys():
    MeanLoss_LR = []
    MeanLoss_RF = []
    MeanLoss_SVR = []
    MeanLoss_GBR = []
    MeanLoss_ANN = []
    MeanLoss_QTP = []
    MeanLoss_HOL = []
    MeanLoss_LES = []

    for threshold in [0, 5, 10, 15]:
        print('Exp: ', exp)
        df = pd.read_csv(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results/MeanLoss('+str(threshold)+').csv')
        MeanLoss_LR.append(df[exp]['LR'])
        MeanLoss_RF.append(df[exp]['LR'])
        MeanLoss_SVR.append(df[exp]['LR'])
        MeanLoss_GBR.append(df[exp]['LR'])
        MeanLoss_ANN.append(df[exp]['LR'])
        MeanLoss_QTP.append(df[exp]['LR'])
        MeanLoss_HOL.append(df[exp]['LR'])
        MeanLoss_LES.append(df[exp]['LR'])

    x = np.arange(len(labels))  # the label locations
    width = 0.20 # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 4*width/2, MeanLoss_LR, width, label='LR')
    rects2 = ax.bar(x - 3*width/2, MeanLoss_RF, width, label='RF')
    rects3 = ax.bar(x - 2*width/2, MeanLoss_SVR, width, label='SVR')
    rects4 = ax.bar(x - width/2, MeanLoss_GBR, width, label='GBR')
    rects5 = ax.bar(x + width/2, MeanLoss_ANN, width, label='ANN')
    rects6 = ax.bar(x + 2*width/2, MeanLoss_QTP, width, label='QTP')
    rects7 = ax.bar(x + 3*width/2, MeanLoss_HOL, width, label='HOL')
    rects8 = ax.bar(x + 4*width/2, MeanLoss_LES, width, label='LES')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Loss')
    ax.set_xlabel('Delta (sec)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.savefig(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results\MeanLoss/Mean_loss_exp_'+dict_exp[exp]+'.png')
