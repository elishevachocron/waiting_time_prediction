import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


path_results = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Synthetic_Data_Generation\GridSearchCV\Good_Split'

path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'

day_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 6: 'Sunday'}


for exp in [1]:
    print('--------------------------------------Experiment ' + str(exp)+'--------------------------------------')
    #Rho
    #
    # df_arrivals = pd.read_csv(path + str(exp) + '/Arrivals.csv')
    # df_service_time = pd.read_csv(path + str(exp) + '/Service_time.csv')
    # df_abandonment = pd.read_csv(path + str(exp) + '/Abandonment.csv')
    # df_number_of_agents = pd.read_csv(path + str(exp) + '/number_of_agents.csv')
    # df_service_time = df_service_time.rename(columns={"private": "s_private", "not_private": "s_not_private"})
    # df_parameters = pd.concat(
    #     [df_arrivals, df_service_time[['s_private', 's_not_private']], df_number_of_agents['number_of_agents']], axis=1)
    #
    # df_parameters['rho'] = (df_parameters['private'] + df_parameters['not_private']) / (
    #             df_parameters['number_of_agents'] * (((df_parameters['private']/df_parameters['s_private']) +
    #                                                    (df_parameters['not_private']/df_parameters['s_not_private'])) / (
    #                                                              df_parameters['private'] + df_parameters[
    #                                                          'not_private'])))/3600
    #
    # df_parameters[['Weekday', 'hour_bin', 'rho']].to_csv(path + str(exp) + '/Rho.csv')

    df_simulator = pd.read_csv(path + str(exp) + '/New_features_simulation_shrinked.csv')
    #Waiting Time Distribution
    data = df_simulator['Real_WT'].to_numpy() * 3600
    plt.figure()
    plt.hist(data, weights=(np.ones_like(data) / len(data)))
    plt.xlabel('Waiting time (sec)')
    plt.ylabel('Pourcentage')
    plt.savefig(path + str(exp) + '\Results_lower_system/Waiting_time_distribution_new_set.png')

    # Waiting Time + QT prediction per hour
    if exp==0:
        x = np.arange(0, 24, 1)
    else:
        x = np.arange(8, 20, 1)
    df_simulator['hour_bin'] = (df_simulator['Arrival_time']).astype(int)
    df_QT_prediction = df_simulator.groupby(['hour_bin']).mean().reset_index()[['WT_QTD_bis', 'LES', 'HOL', 'Real_WT']]
    df_QT_prediction = df_QT_prediction*3600

    plt.figure()
    plt.plot(x, df_QT_prediction['Real_WT'].to_numpy(), label='Average_Real_WT')
    plt.plot(x, df_QT_prediction['HOL'].to_numpy(), label='Average_HOL')
    plt.plot(x, df_QT_prediction['LES'].to_numpy(), label='Average_LES')
    plt.plot(x, df_QT_prediction['WT_QTD_bis'].to_numpy(), label='Average_WT_QTD')
    plt.xlabel('Hour')
    plt.ylabel('Waiting Time (sec)')
    plt.legend()
    plt.savefig(path + str(exp) + '\Results_lower_system/QT_prediction_new_set.png')

    if exp in [0, 3, 9, 10, 11]:

        abandonment = df_simulator.loc[df_simulator.Checkin_time == np.inf].groupby(['hour_bin']).count().reset_index()
        fig, ax1 = plt.subplots()
        s1 = df_QT_prediction['Real_WT'].to_numpy()
        ax1.plot(x, s1, 'b-')
        ax1.set_xlabel('Hour')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('Waiting time (sec)', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        s2 = abandonment['Arrival_num'].to_numpy()
        ax2.plot(x, s2, 'r-')
        ax2.set_ylabel('Number of abandonments', color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        plt.savefig(path + str(exp) + '\Results/Waiting_time_VS_Abandonments_new_set.png')

    # Arrivals_weekday = []
    # Abandonment_weekday = []
    # Nb_agents_weekday = []
    # Service_time_weekday = []
    # Waiting_time_weekday = []
    #
    # for day in [2, 6]: #only wednesday and sunday (highest gap)
    #     print('Day: ', day_dict[day])
    #     Arrival_private = []
    #     Arrival_not_private = []
    #     Abandonment = []
    #     Nb_agents = []
    #     Service_time = []
    #     Waiting_time = []
    #     Nb_of_service_samples = []
    #     for hour_treated in np.arange(0, 12, 1):
    #         print('Hour: ', hour_treated)
    #         df_simulator = pd.read_csv(path+str(exp)+'/Two_months_simulation.csv')
    #         # df_simulator = df_simulator[df_simulator.Week == 0] # First Week
    #         df_simulator = df_simulator[df_simulator.Day == day] # take all the monday for example
    #         df_simulator = df_simulator.dropna(subset=["Real_WT"], axis=0, how='any')
    #         df_simulator['Arrival_time'] = df_simulator['Arrival_time'].astype(int)
    #         df_simulator = df_simulator[df_simulator.Arrival_time == hour_treated]
    #         total_private_arrival = df_simulator[df_simulator.type == 0]
    #         total_not_private_arrival = df_simulator[df_simulator.type == 1]
    #         private_arrivals = int(len(total_private_arrival) / len(df_simulator['Week'].unique()))
    #         not_private_arrivals = int(len(total_not_private_arrival) / len(df_simulator['Week'].unique()))
    #         df_get_service = df_simulator.replace([np.inf, -np.inf], np.nan).dropna(subset=["Service_time"], how="all")
    #         abandonment = int((len(df_simulator) - len(df_get_service)) / len(df_simulator['Week'].unique()))
    #         without_wt = int(len(df_get_service.loc[df_get_service.Real_WT == 0]) / len(df_simulator['Week'].unique()))
    #         with_wt = int(len(df_get_service.loc[df_get_service.Real_WT != 0]) / len(df_simulator['Week'].unique()))
    #         if df_get_service.loc[df_get_service.Real_WT != 0].empty: # if nobody wait
    #             avg_wt = 0
    #         else:
    #             avg_wt = int((df_get_service.loc[df_get_service.Real_WT != 0]['Real_WT'].mean() * 3600))
    #         service_time = int(df_get_service['Service_time'].mean() * 3600)
    #         agents = 0
    #
    #         for day_synthetic in range(len(df_simulator['Week'].unique())):
    #             df_agent = df_simulator.loc[df_simulator['Week'] == day_synthetic]
    #             number_of_agents_daily = np.mean([len(df_agent.loc[(df_agent.Checkin_time < hour_treated + i / 6) & (
    #                                               df_agent.Exit_time > (hour_treated + i / 6))]) for i in range(1, 6)])
    #             agents += number_of_agents_daily
    #
    #         number_of_agents = int(agents / len(df_simulator['Week'].unique()))
    #
    #         Arrival_private.append(private_arrivals)
    #         Arrival_not_private.append(not_private_arrivals)
    #         Abandonment.append(abandonment)
    #         Nb_agents.append(number_of_agents)
    #         Service_time.append(service_time)
    #         Waiting_time.append(avg_wt)
    #         Nb_of_service_samples.append(len(df_get_service))
    #
    #     Arrivals_weekday.append([Arrival_private, Arrival_not_private])
    #     Abandonment_weekday.append(Abandonment)
    #     Nb_agents_weekday.append(Nb_agents)
    #     Service_time_weekday.append(Service_time)
    #     Waiting_time_weekday.append(Waiting_time)
    #
    # print('hello')
    #
    # result_dict = {'Arrival': Arrivals_weekday, 'Abandonment': Abandonment_weekday,
    #                'Nb_agents': Nb_agents_weekday, 'Service_time': Service_time_weekday, 'Waiting_time': Waiting_time_weekday}
    #
    # x= np.arange(8, 20, 1)
    #
    # fig = plt.figure(figsize=(15, 7))
    # ax0 = plt.subplot(2, 2, 1)
    # ax0.plot(x, Arrivals_weekday[0][0], label='Wednesday - Private', color='orange')
    # ax0.plot(x, Arrivals_weekday[0][1], '--', label='Wednesday - Not Private',  color='orange')
    # ax0.plot(x, Arrivals_weekday[1][0], label='Sunday - Private', color='skyblue')
    # ax0.plot(x, Arrivals_weekday[1][1], '--', label='Sunday - Not Private', color='skyblue')
    # ax0.set_xlabel('Hour')
    # ax0.set_ylabel('Number of Arrivals')
    # ax0.legend()
    #
    # ax1 = plt.subplot(2, 2, 2)
    # ax1.plot(x, Abandonment_weekday[0], label='Wednedsay', color='orange')
    # ax1.plot(x, Abandonment_weekday[1], label='Sunday', color='skyblue')
    # ax1.set_xlabel('Hour')
    # ax1.set_ylabel('Number of Abandonments')
    # ax1.legend()
    #
    # ax2 = plt.subplot(2, 2, 3)
    # ax2.plot(x, Nb_agents_weekday[0], label='Wednedsay', color='orange')
    # ax2.plot(x, Nb_agents_weekday[1], label='Sunday', color='skyblue')
    # ax2.set_xlabel('Hour')
    # ax2.set_ylabel('Number of Agents')
    # ax2.legend()
    #
    # ax3 = plt.subplot(2, 2, 4)
    # ax3.plot(x, Service_time_weekday[0], label='Wednedsay', color='orange')
    # ax3.plot(x, Service_time_weekday[1], label='Sunday', color='skyblue')
    # ax3.set_xlabel('Hour')
    # ax3.set_ylabel('Service_time')
    # ax3.legend()
    #
    # plt.suptitle('Parameters')
    # plt.savefig(path+str(exp)+'\Plots_validation/Parameters.png')
    #
    # plt.figure()
    # plt.plot(x, Waiting_time_weekday[0], label='Wednedsay', color='orange')
    # plt.plot(x, Waiting_time_weekday[1], label='Sunday', color='skyblue')
    # plt.xlabel('Hour')
    # plt.ylabel('Waiting_time (sec)')
    # plt.title('Waiting time')
    # plt.legend()
    # plt.savefig(path+str(exp)+'\Plots_validation/Waiting_time.png')
    # print('hello')
    #







        #
        # for result_name, results in result_dict.items():
        #
        #     if result_name == 'Nb_of_samples':
        #         t = np.arange(0, 24, 1)
        #         data1 = diff
        #         data2 = Nb_of_service_samples
        #         fig, ax1 = plt.subplots()
        #         color = 'tab:red'
        #         ax1.set_xlabel('Hour')
        #         ax1.set_ylabel('Difference Service time (sec)', color=color)
        #         ax1.plot(t, data1, color=color)
        #         ax1.tick_params(axis='y', labelcolor=color)
        #         ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #         color = 'tab:blue'
        #         ax2.set_ylabel('Number of samples', color=color)  # we already handled the x-label with ax1
        #         ax2.plot(t, data2, color=color)
        #         ax2.tick_params(axis='y', labelcolor=color)
        #         fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #         plt.title(day_dict[day] + '_' + result_name)
        #         plt.legend()
        #         #plt.savefig(day_dict[day] + '_' + result_name)
        #         plt.show()
        #
        #     else:
        #         plt.figure()
        #         x = np.arange(0, 24, 1)
        #         plt.plot(x, results[0], label=result_name + '_synthetic')
        #         plt.plot(x, results[1], label=result_name + '_real')
        #         if result_name == 'Waiting_time':
        #             plt.plot(x, results[2], label='Confidence_interval', color='red')
        #             plt.plot(x, results[3], label='Confidence_interval', color='red')
        #         plt.xlabel('Hour')
        #         plt.xticks(np.arange(0, 24, 1), np.arange(0, 24, 1))
        #         plt.ylabel(result_name)
        #         plt.title(day_dict[day] + '_' + result_name)
        #         plt.legend()
        #         #plt.savefig(day_dict[day] + '_' + result_name)
        #         plt.show( )
        #
