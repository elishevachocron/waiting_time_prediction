import pandas as pd
import numpy as np
import time
import socket


if __name__ == '__main__':
    start = time.time()

    for exp in [0]:
        print('Experiment: ', exp)
    hostname = socket.gethostname()

    if hostname == 'DESKTOP-A925VLR':
        path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'
    else:
        path = r'C:\Users\elishevaz\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'

    file_name = '/Simulation_corrected_full_imple.csv'
    df_simulator = pd.read_csv(path + str(exp) + file_name)
    df = df_simulator.loc[:, ~df_simulator.columns.str.contains('^Unnamed')]

    for idx in range(len(df)):
        if df.loc[idx, 'Arrival_time'] < 8.75:  # to correct only after the first half hour
            continue
        if idx % 10000 == 0:
            print('Correction: ' + str(round((idx / len(df) * 100), 1)) + ' %')

        previous_days = df.loc[(df['Week'] < df.loc[idx]['Week']) & (df['Day'] < df.loc[idx]['Day'])]
        curr_day = df.loc[(df['Week'] == df.loc[idx]['Week']) & (df['Day'] == df.loc[idx]['Day'])]
        relevant_curr_day = curr_day.loc[(curr_day['Arrival_time'] < df.loc[idx]['Arrival_time']) & (curr_day['Exit_time'] < df.loc[idx]['Arrival_time'])]
        relevant = pd.concat([previous_days, relevant_curr_day], ignore_index=True)

        #parameters computation
        mu = 1 / relevant['Service_time'].mean()
        mu_private = 1 / relevant.loc[relevant.type == 0]['Service_time'].mean()
        mu_not_private = 1 / relevant.loc[relevant.type == 1]['Service_time'].mean()

        df.loc[idx, 'mu'] = mu
        df.loc[idx, 'mu_private'] = mu_private
        df.loc[idx, 'mu_not_private'] = mu_not_private
        df.loc[idx, 'previous_WT'] = relevant['Real_WT'].sum()
        df.loc[idx, 'abandonment'] = (relevant['Checkin_time'] == np.inf).sum()


        if df.loc[idx, 'n_available'] > 0:
            df.loc[idx, 'QTP_data_based'] = 0
            df.loc[idx, 'QTP_weighted'] = 0
            df.loc[idx, 'QTP_weighted_bis'] = 0

        else:
            df.loc[idx, 'QTP_data_based'] = np.sum(1 / (
                        df.loc[idx, 'n_servers'] * df.loc[idx, 'mu'] + df.loc[
                    idx, 'theta'] * np.arange(
                    df.loc[idx, 'queue1'] + df.loc[idx, 'queue2'] + 1)))

            if df.loc[idx, 'total_queue'] == 0:
                mu_weighted = mu
                df.loc[idx, 'QTP_weighted_bis'] = np.sum(1 /
                                                         (df.loc[idx, 'n_servers'] * mu_weighted +
                                                          df.loc[idx, 'theta'] * np.arange(df.loc[idx, 'total_queue'] + 1)))

            else:
                mu_weighted = float((df.loc[idx, 'queue1'] * mu_private + df.loc[
                    idx, 'queue2'] * mu_not_private) / df.loc[idx, 'total_queue'])
                mu_weighted_bis = float((df.loc[idx, 'queue1'] / mu_private) + (
                            df.loc[idx, 'queue2'] / mu_not_private))
                df.loc[idx, 'QTP_weighted_bis'] = np.sum(1 / (((df.loc[idx, 'n_servers'] *
                                                                               df.loc[idx, 'total_queue']) / mu_weighted_bis) +
                                                                             df.loc[idx, 'theta'] * np.arange(
                            df.loc[idx, 'total_queue'] + 1)))

            df.loc[idx, 'mu_weighted'] = mu_weighted

            df.loc[idx, 'QTP_weighted'] = np.sum(
                1 / (df.loc[idx, 'n_servers'] * mu_weighted +
                     df.loc[idx, 'theta'] * np.arange(df.loc[idx, 'total_queue'] + 1)))

    end = time.time()
    print('Time elapsed: ', int((end - start) / 60))
    df.to_csv(path + str(exp) + '/Arik_features_imple.csv')
