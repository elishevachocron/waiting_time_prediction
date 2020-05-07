import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Data Exploration\Excel files\Simulator Data\One_month_exploration/'

arr_rates_types = np.zeros((2, 24), dtype=int)
df_arrivals = pd.read_csv(path + 'Arrivals.csv')
df_service_time = pd.read_csv(path + 'Service_time.csv')
df_abandonment = pd.read_csv(path + 'Abandonment.csv')
df_number_of_agents = pd.read_csv(path + 'number_of_agents.csv')

data = {'Arrivals': df_arrivals, 'Service_time': df_service_time, 'Abandonment': df_abandonment, 'number_of_agents': df_number_of_agents}
weekday = {0: ('Monday', []), 1: ('Tuesday', []), 2: ('Wednesday', []), 3: ('Thursday', []), 6: ('Sunday', [])}


#Arrivals
average_arrivals = []
plt.figure()
x = np.arange(0, 24, 1)
for day, tuple in weekday.items():
    arrivals = df_arrivals.loc[df_arrivals.Weekday == day]['private'] + df_arrivals.loc[df_arrivals.Weekday == day]['not_private']
    average_arrivals.append(np.mean(arrivals[8:20]))
    plt.plot(x, arrivals, label=tuple[0])
plt.xlabel('Hour')
plt.xticks(np.arange(0, 24, 1), np.arange(0, 24, 1))
plt.ylabel('Arrivals')
plt.title('Arrivals Distribution')
plt.legend()
#plt.savefig('Arrivals_Distribution.png')

#Service_rate
average_service_rate_private = []
average_service_rate_not_private = []
plt.figure()
x = np.arange(0, 24, 1)
for day, tuple in weekday.items():
    service_time_private = df_service_time.loc[df_service_time.Weekday == day]['private']
    service_time_not_private = df_service_time.loc[df_service_time.Weekday == day]['not_private']
    average_service_rate_private.append(np.mean(service_time_private[8:20]))
    average_service_rate_not_private.append(np.mean(service_time_not_private[8:20]))
    plt.plot(x, service_time_private, label=tuple[0])
plt.xlabel('Hour')
plt.xticks(np.arange(0, 24, 1), np.arange(0, 24, 1))
plt.ylabel('Service time')
plt.title('Service time Distribution')
plt.legend()
#plt.savefig('Service_time_Distribution.png')

#Number of agents
average_nb_agents = []
plt.figure()
x = np.arange(0, 24, 1)
for day, tuple in weekday.items():
    nb_agents = df_number_of_agents.loc[df_number_of_agents.Weekday == day]['number_of_agents']
    average_nb_agents.append(np.mean(nb_agents[8:20]))
    plt.plot(x, nb_agents, label=tuple[0])
plt.xlabel('Hour')
plt.xticks(np.arange(0, 24, 1), np.arange(0, 24, 1))
plt.ylabel('Number of agents')
plt.title('Number of agents Distribution')
plt.legend()
#plt.savefig('Nb_of_agents_Distribution.png')

#Abandonments

plt.figure()
average_abandonment = []
x = np.arange(0, 24, 1)
for day, tuple in weekday.items():
    abandonment = df_abandonment.loc[df_abandonment.Weekday == day]['wait_time']
    average_abandonment.append(np.mean(abandonment[8:20]))
    plt.plot(x, abandonment, label=tuple[0])
plt.xlabel('Hour')
plt.xticks(np.arange(0, 24, 1), np.arange(0, 24, 1))
plt.ylabel('Abandonments - Patience')
plt.title('Abandonments Distribution')
plt.legend()
#plt.savefig('Abandonments_Distribution.png')

df = pd.DataFrame({'Mean': [np.mean(average_arrivals), np.mean(average_service_rate_private), np.mean(average_service_rate_not_private), np.mean(average_nb_agents), np.mean(average_abandonment)],
                   'STD': [np.std(average_arrivals), np.std(average_service_rate_private), np.std(average_service_rate_not_private), np.std(average_nb_agents), np.std(average_abandonment)]},
                  index=['Arrivals', 'Service_rate_private', 'Service_rate_not_private', 'Number_of_agents', 'Abandonment'])

df.to_csv(r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Weekday Comparison\Mean+std.csv')