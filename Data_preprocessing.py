import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings
import time

start = time.time()
warnings.filterwarnings('ignore')
# import MMn_class as data

# Download the data from the MMn_class
# data = data.df


data = pd.read_csv(r'C:\Users\Elisheva\Desktop\Two_months_simulation.csv')


data.loc[data.Real_WT == np.inf, 'Real_WT'] = data.loc[data.Real_WT == np.inf, 'Exit_time'] - data.loc[data.Real_WT == np.inf, 'Arrival_time'] #replace the abandonment cases
#data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Real_WT'], axis=0, how="any")# remove all the missing value
# Defining the specific features we want to work with
data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
df = data[['Arrival_time', 'n_servers', 'LSD', 'queue1', 'queue2', 'Week', 'Day', 'WT_QTD', 'LES', 'HOL', 'Real_WT']]
df = data
features_type = {'features_type_1': ['Arrival_time', 'n_servers', 'LSD', 'queue1', 'queue2', 'Day']}

# ---------------Test/Train Split----------------
for key_type, features_list in features_type.items():
    print('Test/Train Split')
    # QT estimation of Waiting Time
    y_QTD_test = df.loc[df.Week > 7]['WT_QTD'].to_frame()
    y_LES_test = df.loc[df.Week > 7]['LES'].to_frame()
    y_HOL_test = df.loc[df.Week > 7]['HOL'].to_frame()
    # Real Waiting Time
    y = df['Real_WT'].to_frame()
    X_train = df.loc[df.Week <= 7][features_list]
    X_test = df.loc[df.Week > 7][features_list]
    y_train = df.loc[df.Week <= 7]['Real_WT'].to_frame()
    y_test = df.loc[df.Week > 7]['Real_WT'].to_frame()

    # X = df.loc[:, df.columns.difference(['Checkin_time', 'Service_time', 'Exit_time', 'WT_QTD', 'Real_WT', 'LES', 'HOL'])]
    #
    # X_train, X_test, y_train, y_test, y_QTD_train, y_QTD_test, y_LES_train, y_LES_test, \
    # y_HOL_train, y_HOL_test = train_test_split(X, y, y_QTD, y_LES, y_HOL, test_size=0.30, random_state=42)

    # print('Saving')
    # dict_split = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'y_QTD_test': y_QTD_test,
    #               'y_LES_test': y_LES_test, 'y_HOL_test': y_HOL_test}
    #
    # for key, value in dict_split.items():
    #     value.to_csv(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Synthetic_Data_Generation\Good Split/' \
    #                 + str(key) + '.csv', index=False)

end = time.time()

print('Preprocessing - Time elapsed: ', int((end-start)))