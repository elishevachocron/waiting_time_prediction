import pandas as pd
import numpy as np
import time
import multiprocessing

if __name__ == '__main__':
    WINDOW = 100
    computer = input("Izik's Computer?: ")

    if computer == '0':
        path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'
    else:
        path = r'C:\Users\elishevaz\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'

    df = pd.DataFrame()

    for exp in [9]:
        print('--------------------------------------Experiment ' + str(exp)+'--------------------------------------')
        df_simulator = pd.read_csv(path + str(exp) + '/New_features_simulation_shrinked.csv')
        for week in df_simulator['Week'].unique():
            for day in df_simulator['Day'].unique():
                start = time.time()
                print('Week: ' + str(week) +' Day: ' + str(day))
                df_current_day = df_simulator.loc[(df_simulator.Week == week) & (df_simulator.Day == day)]

                for index in np.arange(len(df), len(df)+len(df_current_day), 1):
                    if index % 500 == 0:
                        print('Index: ', index)
                    relevant_lines = df_current_day.loc[(df_current_day['Checkin_time'] < df_current_day.loc[index]['Arrival_time']) | (df_current_day['Exit_time'] < df_current_day.loc[index]['Arrival_time'])].iloc[-WINDOW:]

                    if len(relevant_lines) > 0:
                        df_current_day.loc[index, 'abandonments'] = (relevant_lines['Checkin_time'] == np.inf).sum()
                        df_current_day.loc[index, 'previous_WT'] = relevant_lines['Real_WT'].sum()
                        if df_current_day.loc[index, 'previous_WT'] == 0:
                            df_current_day.loc[index, 'teta'] = 0
                        else:
                            df_current_day.loc[index, 'teta'] = df_current_day.loc[index, 'abandonments'] / df_current_day.loc[index, 'previous_WT']

                        if df_current_day.loc[index, 'n_available'] != 0:
                            df_current_day.loc[index, 'WT_QTD_bis'] = 0
                        else:
                            df_current_day.loc[index, 'WT_QTD_bis'] = np.sum(
                                1 / (df_current_day.loc[index, 'n_servers'] * df_current_day.loc[index, 'mu']
                                     + df_current_day.loc[index, 'teta'] * np.arange(
                                            df_current_day.loc[index, 'queue1'] + df_current_day.loc[index, 'queue2'] + 1)))
                    else:
                        df_current_day.loc[index, 'abandonments'] = 0
                        df_current_day.loc[index, 'previous_WT'] = 0
                        df_current_day.loc[index, 'teta'] = 0
                        df_current_day.loc[index, 'WT_QTD_bis'] = 0


                df = pd.concat([df, df_current_day], ignore_index=True)
                end = time.time()
                print('Time elapsed: ', int((end-start)/60))
                df.to_csv(path + str(exp) + '/test.csv')
        df.to_csv(path + str(exp) + '/New_features_simulation_shrinked.csv')


        if df['LES'].isna().sum() == 0:
            print('No need of LES correction')
        else:
            print('LES correction in progress...')
            index_to_correct = np.where(np.isnan(df['LES'].to_numpy()))[0]
            print('Missing data: ', len(index_to_correct))
            for iter, index in enumerate(index_to_correct):
                if iter == 10000:
                    print('10 000 have been done')
                if iter == 20000:
                    print('20 000 have been done')
                df_possibilities = df.loc[(df.Week == df.iloc[index]['Week']) &
                                                    (df.Day == df.iloc[index]['Day']) &
                                                    (df.Checkin_time < df.iloc[index]['Arrival_time'])]
                idx_les = df_possibilities['Checkin_time'].idxmax()
                df.loc[df.index == index, 'LES'] = df['Real_WT'].iloc[idx_les]
            if df['LES'].isna().sum() == 0:
                print('Done')
                df.to_csv(path + str(exp) + '/New_features_simulation_shrinked.csv')
            else:
                print('There is a pb with the LES correction')



