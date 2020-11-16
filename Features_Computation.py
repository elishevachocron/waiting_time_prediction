import pandas as pd
import numpy as np
import time
import math
import socket
import warnings


warnings.filterwarnings('ignore')
#########################################################################################################################
full_comp = 0 #flag for full computation (the whole feature list)

if full_comp:
    to_implement = ['LES', 'HOL', 'pack_relevant_based', 'in_service_p', 'in_service_np']
else:
    to_implement = ['pack_relevant_based',]

correction = 0
#########################################################################################################################

#----------------------------------------------------FUNCTION DEFINITION----------------------------------------------------

def LES_comp(df, idx):
    if idx != 0:
        got_service = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Checkin_time'], axis=0, how="any")
        got_service.reset_index(drop=True, inplace=True)
        les_idx = got_service.loc[(got_service['Checkin_time'] < df.loc[idx]['Arrival_time'])]['Checkin_time'].values.argmax()
        les_WT = got_service.loc[les_idx]['Real_WT']
    else:
        les_WT = 0

    return les_WT

def HOL_comp(df, idx):
    queue = df.loc[(df['Arrival_time'] < df.loc[idx]['Arrival_time']) & (df['Checkin_time'] > df.loc[idx]['Arrival_time'])
                   & (df['Exit_time'] > df.loc[idx]['Arrival_time'])]
    queue.reset_index(drop=True, inplace=True)
    if queue.empty:#There is no queue
        hol_WT = 0
    else:
        hol_idx = queue['Arrival_time'].values.argmin()
        hol_WT = df.loc[idx]['Arrival_time'] - queue.loc[hol_idx]['Arrival_time']
    return hol_WT

def in_service_p_comp (df, idx):
    private = len(df.loc[(df['Checkin_time'] < df.loc[idx]['Arrival_time']) & (df['Exit_time'] > df.loc[idx]['Arrival_time']) & (df['type'] == 0)])
    return private

def in_service_np_comp (df, idx):
    not_private = len(df.loc[(df['Checkin_time'] < df.loc[idx]['Arrival_time']) & (df['Exit_time'] > df.loc[idx]['Arrival_time']) & (df['type'] == 1)])
    return not_private

def relevant_df_comp (df, idx):

    relevant = df.loc[(df['Arrival_time'] < df.loc[idx]['Arrival_time']) & (df['Arrival_time'] > float(df.loc[idx]['Arrival_time']-0.75)) & (df['Exit_time'] < df.loc[idx]['Arrival_time'])]
    mu = 1 / relevant['Service_time'].mean()
    mu_private = 1 / relevant.loc[relevant.type == 0]['Service_time'].mean()
    mu_not_private = 1 / relevant.loc[relevant.type == 1]['Service_time'].mean()
    previous_WT = relevant['Real_WT'].sum()
    abandonment = (relevant['Checkin_time'] == np.inf).sum()

    if abandonment == 0:
        theta = 0
    else:
        theta = (relevant['Checkin_time'] == np.inf).sum()/relevant['Real_WT'].sum()

    return mu, mu_private, mu_not_private, theta, previous_WT, abandonment


if __name__ == '__main__':

    hostname = socket.gethostname()

    if hostname == 'DESKTOP-A925VLR':
        path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'
    else:
        path = r'C:\Users\elishevaz\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'

    df = pd.DataFrame()
    dict_func_features = {'LES': LES_comp, 'HOL': HOL_comp,
                          'in_service_p': in_service_p_comp,
                          'in_service_np': in_service_np_comp,
                          'pack_relevant_based': relevant_df_comp}

    for exp in [1, 3, 9]:
        print('--------------------------------------Experiment ' + str(exp)+'--------------------------------------')
        beginning = 0.75
        df = pd.DataFrame() # the final dataframe

        if full_comp: # compute all the features
            file_name = '/New_features_simulation_lowest_system.csv'
            df_simulator = pd.read_csv(path + str(exp) + file_name)
            simulator_features = ['Arrival_time', 'Checkin_time', 'Service_time', 'Exit_time', 'type',
                                  'n_servers', 'n_available', 'n_not_available', 'LSD', 'queue1', 'queue2',
                                   'Week', 'Day', 'Real_WT']

            data = df_simulator[simulator_features]
            data['total_queue'] = data['queue1'] + data['queue2']

        else: #compute several features
            file_name = '/Simulation_corrected_full_imple.csv'
            df_simulator = pd.read_csv(path + str(exp) + file_name)
            data = df_simulator



        if exp == 0 or exp == 12:#removing the night's hours
            data = data.loc[(data.Arrival_time >= 8) & (data.Arrival_time < 20)]
            data.reset_index(drop=True, inplace=True)
            beginning = 8.75

        if correction:
            data.loc[(data.Arrival_time > beginning) & (data.total_queue == 0) &
                     (data.n_available == 0), 'QTP_weighted_bis'] = data.loc[(data.Arrival_time > beginning) &
                                                                             (data.total_queue == 0) & (data.n_available == 0), 'QTP_weighted']
            data.to_csv(path + str(exp) + '/Simulation_corrected_full_imple_bis.csv')
            continue

        for week in data['Week'].unique():
            for day in data['Day'].unique():
                start = time.time()
                print('Week: ' + str(week) +' Day: ' + str(day))
                df_current_day = data.loc[(data.Week == week) & (data.Day == day)]
                df_current_day.reset_index(drop=True, inplace=True)

                for df_idx in range(len(df_current_day)):
                    if df_current_day.loc[df_idx, 'Arrival_time'] < beginning: # to correct only after the first half hour
                        continue
                    if df_idx % 500 == 0:
                        print('Correction: ' + str(round((df_idx/len(df_current_day)*100), 1)) + ' %')
                    for feature in to_implement:
                        if feature != 'pack_relevant_based':
                            df_current_day.loc[df_idx, feature] = dict_func_features[feature](df_current_day, df_idx)
                        elif feature == 'pack_relevant_based':
                            mu, mu_private, mu_not_private, theta, previous_WT, abandonment = dict_func_features[feature](df_current_day[:df_idx+1], df_idx)
                            df_current_day.loc[df_idx, 'mu_private'] = mu_private
                            df_current_day.loc[df_idx, 'mu_not_private'] = mu_not_private
                            df_current_day.loc[df_idx, 'mu'] = mu
                            df_current_day.loc[df_idx, 'theta'] = theta
                            df_current_day.loc[df_idx, 'previous_WT'] = previous_WT
                            df_current_day.loc[df_idx, 'abandonment'] = abandonment

                        #Queueing Theory Predictors (QTP)

                            if df_current_day.loc[df_idx, 'n_available'] > 0:
                                df_current_day.loc[df_idx, 'QTP_data_based'] = 0
                                df_current_day.loc[df_idx, 'QTP_weighted'] = 0
                                df_current_day.loc[df_idx, 'QTP_weighted_bis'] = 0

                            else:
                                df_current_day.loc[df_idx, 'QTP_data_based'] = np.sum(1 / (df_current_day.loc[df_idx, 'n_servers'] * df_current_day.loc[df_idx, 'mu'] + df_current_day.loc[df_idx, 'theta'] * np.arange(
                                                                df_current_day.loc[df_idx, 'queue1'] + df_current_day.loc[df_idx, 'queue2'] + 1)))

                                if df_current_day.loc[df_idx, 'total_queue'] == 0:
                                    mu_weighted = mu
                                    df_current_day.loc[df_idx, 'QTP_weighted_bis'] = np.sum(1 / (df_current_day.loc[df_idx, 'n_servers'] * mu_weighted +
                                                                                     df_current_day.loc[df_idx, 'theta'] * np.arange(df_current_day.loc[df_idx, 'total_queue'] + 1)))

                                else:
                                    mu_weighted = float((df_current_day.loc[df_idx, 'queue1'] * mu_private + df_current_day.loc[
                                        df_idx, 'queue2'] * mu_not_private) / df_current_day.loc[df_idx, 'total_queue'])
                                    mu_weighted_bis = float((df_current_day.loc[df_idx, 'queue1'] / mu_private) + (df_current_day.loc[df_idx, 'queue2'] / mu_not_private))
                                    df_current_day.loc[df_idx, 'QTP_weighted_bis'] = np.sum(1 / (((df_current_day.loc[df_idx, 'n_servers'] * df_current_day.loc[df_idx, 'total_queue']) / mu_weighted_bis) + df_current_day.loc[df_idx, 'theta'] * np.arange(df_current_day.loc[df_idx, 'total_queue'] + 1)))

                                df_current_day.loc[df_idx, 'mu_weighted'] = mu_weighted

                                df_current_day.loc[df_idx, 'QTP_weighted'] = np.sum(1 / (df_current_day.loc[df_idx, 'n_servers'] * mu_weighted +
                                                                                     df_current_day.loc[df_idx, 'theta'] * np.arange(df_current_day.loc[df_idx, 'total_queue'] + 1)))


                df = pd.concat([df, df_current_day], ignore_index=True)
                end = time.time()
                print('Time elapsed: ', int((end-start)/60))

        df.to_csv(path + str(exp) + '/Reduced_system_full_imple.csv')


