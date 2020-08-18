import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set()

metrics = ['RE', 'RRMSE']
labels = ['1.0', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '3.0', '3.1', '3.2', '4']
#labels = ['1.0', '2.0', '2.4', '2.5']
path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments'
QTP_RRMSE = []
RF_RRMSE = []
ANN_RRMSE = []

QTP_RE = []
RF_RE = []
ANN_RE = []

for exp in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]:
#for exp in [1, 2, 6, 7]:
    print('Exp: ', exp)
    df = pd.read_csv(path + '\Experiment_'+str(exp)+'\Results/Results_new_got_service_set_features_1.csv')
    QTP_RRMSE.append(float(df.loc[df['Unnamed: 0'] == 'QTD']['RRMSE_test']))
    RF_RRMSE.append(float(df.loc[df['Unnamed: 0'] == 'RF']['RRMSE_test']))
    ANN_RRMSE.append(float(df.loc[df['Unnamed: 0'] == 'ANN']['RRMSE_test']))
    QTP_RE.append(float(df.loc[df['Unnamed: 0'] == 'QTD']['RE']))
    RF_RE.append(float(df.loc[df['Unnamed: 0'] == 'RF']['RE']))
    ANN_RE.append(float(df.loc[df['Unnamed: 0'] == 'ANN']['RE']))

dict_df = {'RE': [QTP_RE, RF_RE, ANN_RE, 'Relative Error'], 'RRMSE': [QTP_RRMSE, RF_RRMSE, ANN_RRMSE, 'Relative RMSE']}

for metric in metrics:
    QTP = dict_df[metric][0]
    RF = dict_df[metric][1]
    ANN = dict_df[metric][2]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, QTP, width, label='QTP')
    rects2 = ax.bar(x , RF, width, label='RF')
    rects3 = ax.bar(x + width , ANN, width, label='ANN')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(dict_df[metric][3])
    ax.set_xlabel('Experiment')
    ax.set_title(metric+' results for each experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    #autolabel(rects1)
    #autolabel(rects2)

    fig.tight_layout()
    plt.savefig(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\General results/'+ metric +'_got_service.png')
    #plt.show()



# results_reduced = pd.read_csv(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Reports/Results_reduced_systems.csv')
#
# metrics = ['RE', 'CV']
# dict_df = {'RE': ['RE (QTP)', 'RE (RF)', 'RE (ANN)', 'Relative Error'], 'CV': ['CV (QTP)', 'CV (RF)', 'CV (ANN)', 'Relative RMSE']}
# labels = ['1.0', '1.0.1', '2.1', '2.1.1', '3.0', '3.0.1']
# for metric in metrics:
#     QTP = results_reduced.loc[results_reduced['Metric'] == dict_df[metric][0]].values[0][1:]
#     RF = results_reduced.loc[results_reduced['Metric'] == dict_df[metric][1]].values[0][1:]
#     ANN = results_reduced.loc[results_reduced['Metric'] == dict_df[metric][2]].values[0][1:]
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.25  # the width of the bars
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width, QTP, width, label='QTP')
#     rects2 = ax.bar(x , RF, width, label='RF')
#     rects3 = ax.bar(x + width , ANN, width, label='ANN')
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel(dict_df[metric][3])
#     ax.set_xlabel('Experiment')
#     ax.set_title(metric+' results for each experiment')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()
#
#
#     def autolabel(rects):
#         """Attach a text label above each bar in *rects*, displaying its height."""
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate('{}'.format(height),
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom')
#
#
#     #autolabel(rects1)
#     #autolabel(rects2)
#
#     fig.tight_layout()
#     plt.savefig(r'C:\Users\Elisheva\Desktop\Master\These\Thesis files/'+ metric +'_reduced_systems.png')
#     #plt.show()