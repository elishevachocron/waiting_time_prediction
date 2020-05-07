import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

plots = []

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

path = 'C:\\Users\\Elisheva\\Dropbox\\QueueMining\\WorkingDocs\\Simulator\\Synthetic_Data_Generation\\GridSearchCV'

features_type = {'features_type_1': ['Arrival_time', 'n_servers', 'LSD', 'queue1', 'queue2', 'Day']}
summary = {}

for key_type, features_list in features_type.items():
    print('-------------------------------------- Type_features: ' + str(key_type) + ' ----------------------------------')

    with open(path + '\\' +'results_dict_' + str(key_type) + '.p', 'rb') as fp:
        data = pickle.load(fp)

    rmse_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    mae_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    cv_metrics = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    hyper_parameters = {'RF': [], 'SVR': [], 'GBR': [], 'ANN': [], 'QTD': [], 'LES': [], 'HOL': []}
    labels = []
    df_results = pd.DataFrame(columns=['RMSE', 'MAE', 'CV'], index=['RF', 'SVR', 'GBR', 'ANN', 'QTD', 'LES', 'HOL'])


    for algo, results in data.items():
        hyper_parameters[algo].append(results[0])
        rmse_metrics[algo].append(results[1])
        mae_metrics[algo].append(results[2])
        cv_metrics[algo].append(results[3])

    df_results['RMSE'] = np.concatenate(list(rmse_metrics.values()))
    df_results['MAE'] = np.concatenate(list(mae_metrics.values()))
    df_results['CV'] = np.concatenate(list(cv_metrics.values()))
    df_results.to_csv(r'C:\Users\Elisheva\Dropbox\QueueMining\WorkingDocs\Simulator\Synthetic_Data_Generation\GridSearchCV/Results.csv')

    METRICS = {'RMSE': rmse_metrics, 'MAE': mae_metrics, 'CV': cv_metrics}

    sum_rmse = pd.DataFrame(data=np.array([list for list in rmse_metrics.values()]), \
                              index=np.array([index for index in cv_metrics.keys()]), columns=labels)
    sum_mae = pd.DataFrame(data=np.array([list for list in mae_metrics.values()]), \
                              index=np.array([index for index in cv_metrics.keys()]), columns=labels)
    sum_cv = pd.DataFrame(data=np.array([list for list in cv_metrics.values()]),\
                           index= np.array([index for index in cv_metrics.keys()]) , columns=labels)

    summary[key_type] = {'RMSE': sum_rmse, 'MAE': sum_mae, 'CV': sum_cv}

    x = np.arange(len(range(1)))  # the label locations
    width = 0.09  # the width of the bars

    for metric_name, list in METRICS.items():
        RF_metric = list['RF']
        SVR_metric = list['SVR']
        GBR_metric = list['GBR']
        ANN_metric = list['ANN']
        QTD_metric = list['QTD']
        LES_metric = list['LES']
        HOL_metric = list['HOL']

        fig, ax = plt.subplots()

        rects1 = ax.bar(x - 3*width, RF_metric, width, label='RF')
        rects2 = ax.bar(x - 2*width, SVR_metric, width, label='SVR')
        rects3 = ax.bar(x - width, GBR_metric, width, label='GBR')
        rects4 = ax.bar(x, ANN_metric, width, label='ANN')
        rects5 = ax.bar(x + width, QTD_metric, width, label='QTD')
        rects6 = ax.bar(x + 2*width, LES_metric, width, label='LES')
        rects7 = ax.bar(x + 3*width, HOL_metric, width, label='HOL')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(str(metric_name))
        ax.set_title(str(metric_name) + ' - Test set delays')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plots.append(ax)
        # autolabel(rects1)
        # autolabel(rects2)
        # autolabel(rects3)
        # autolabel(rects4)
        # autolabel(rects5)
        # autolabel(rects6)
        # autolabel(rects7)

        figure_name = 'Plot_' + str(metric_name) + '_' + str(key_type) + '.png'
        fig.tight_layout()
        plt.savefig(path + '/' + figure_name)
        #plt.show()

for type, dict_metrics in summary.items():
    for metric, results in dict_metrics.items():
        results.to_csv(path + '/' + str(type) + '_' +str(metric) +'.csv')

print('Hello')
