import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set()
#metrics_to_check = ['RE_test', 'RRMSE_test', 'RMeanLoss_10']
metrics_to_check = ['RMSE_test']
#methods_to check = ['QTP', 'QTP_weighted_bis', 'HOL', 'LES', 'LR', 'RF', 'SVR', 'GBR', 'ANN']

results = ['Reduced_systems']

#results = ['Benchmark', 'Customer_influence']

for result_to_check in results:
    print('-------------------------------------- '+ result_to_check + ' -------------------------------------- ')

    if result_to_check == 'Benchmark':
        # labels = ['1.0', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '3.0', '3.1', '3.2', '4.0']
        # label_dict = {1: '1.0', 2: '2.0', 3: '2.1', 4: '2.2', 5: '2.3', 6: '2.4', 7: '2.5', 8: '2.6', 9: '3.0', 10: '3.1', 11: '3.2', 0: '4.0'}
        labels = ['Baseline', 'Abandonment', 'Unbalanced Customer', 'Weekday Variation', 'Combination 1', 'Combination 3', 'Real System']
        label_dict = {1: '1.0', 3: '2.1', 4: '2.2', 7: '2.5', 9: '3.0', 11: '3.2', 0: '4.0'}
    elif result_to_check == 'Customer_influence':
        labels = ['Real System', 'Customer Influence']
        label_dict = {0: '4.0', 12: '4.1'}
    elif result_to_check == 'Reduced_systems':
        labels = ['Baseline', 'R_Baseline', 'Abandonment', 'R_Abandonment', 'Combination 1', 'R_Combination 1']
        label_dict = {1: '1.0', 13: '1.0.1', 3: '2.1', 14: '2.1.1', 9: '3.0', 15: '3.0.1'}

    path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments'

    for metric in metrics_to_check:
        print('-------------------------------------- ' + metric + ' -------------------------------------- ')
        dict_metrics = {'QTP': [], 'QTP_weighted_bis': [], 'HOL': [], 'LES': [], 'LR': [], 'RF': [], 'SVR': [], 'GBR': [], 'ANN': []}
        color_dict = {'QTP': 'darkorange', 'QTP_weighted_bis': 'cornflowerblue', 'HOL': 'indianred', 'LES': 'g', 'LR': 'm', 'RF': 'palevioletred', 'SVR': 'b', 'GBR': 'c', 'ANN': 'sandybrown'}


        for exp in label_dict.keys():
            #print('Exp: ', exp)
            if result_to_check == 'Reduced_systems':
                pass
            df = pd.read_csv(path + '\Experiment_'+str(exp)+'\Final_results/Results_quantile_1.csv')
            for key_metric, metrics_list in dict_metrics.items():
                dict_metrics[key_metric].append(float(df.loc[df['Unnamed: 0'] == key_metric][metric]))


        x = np.arange(len(labels))  # the label locations
        width = 0.15  # the width of the bars

        method_sets = {'QT_set': ['QTP', 'QTP_weighted_bis', 'HOL', 'LES'], 'ML_set': ['LR', 'RF', 'SVR', 'GBR', 'ANN'],
                        'QTvsML_set': ['QTP', 'LR', 'RF', 'ANN'], 'SnapVSML_set': ['HOL', 'LES', 'LR', 'RF', 'ANN']}

        for set, methods_for_plot in method_sets.items():
            print('----------------------------------- ' + set + ' ----------------------------------- ')

            fig, ax = plt.subplots()
            n_plots = len(methods_for_plot)
            half = int(n_plots / 2)
            for i in range(n_plots):
                print('Method:', methods_for_plot[i])
                rects = ax.bar(x - (i - half) * width, dict_metrics[methods_for_plot[i]], width, label= methods_for_plot[i], color= color_dict[methods_for_plot[i]])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(metric)
            ax.set_xlabel('Experiment')
            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35)
            ax.legend()

            fig.tight_layout()
            plt.savefig(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results\Final_results' + '\\' + result_to_check + '\\' + metric + '/' + set + '.png')
            #plt.show()

