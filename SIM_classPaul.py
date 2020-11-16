
import numpy as np
import pandas as pd
import math
import time
import warnings
import math


warnings.filterwarnings('ignore')
#start = time.time()


# In[4]:


def gen_arrivals(arr_rates, mean_patience, total_hours):
    arrs = pd.DataFrame(columns=["time", "patience", "type"])
    arr_nums = np.random.poisson(arr_rates)
    num_types = len(arr_rates)
    for i in range(num_types):
        for j in range(total_hours):
            arr_dict = {'time': j+np.random.uniform(0, 1, arr_nums[i][j]),
                     'patience': np.random.exponential(mean_patience[i][j], arr_nums[i][j]),
                     'type':[i]*arr_nums[i][j]}
            arrs = arrs.append(pd.DataFrame(arr_dict), ignore_index=True)
            #print(arr_nums[i][j])
            #print(arr_dict)
            #print(arrs)
    arrs = arrs.sort_values(by='time')
    arrs = arrs.reset_index(drop=True)
    arrs.insert(0, "Arrival_num", list(arrs.index))
    return arrs
            

class Simulator():
    # arrivals (df: num_arrivals*4)- Arrival_num, time, patience,type
    # week (scalar) - week number
    # day (scalar)  - day of week, between 0 and 5 
    # server_schedule (df: max_servers*24) - 0/1 matrix indictaing which servers available each hour
    # service_mean_types (array 3*1) - mean service times for each customer type (fixed over hours)
    # service_rates (array 24*1) = 1/(mean service time), where mean is weighted over types arriving in that hour

    def __init__(self, arrivals, week, day, server_schedule, service_mean_types, server_rates, total_hours, service_distribution, speed_up_flag):

        self.week = week
        self.day = day
        # arrivals
        self.arrivals = arrivals
        self.num_arrivals = len(self.arrivals)
        # number of customer types
        self.cust_types = len(service_mean_types) # = number of different customer types
        # servers schedule and service rates (the latter by types)
        self.service_distribution = service_distribution
        self.speed_up_flag = speed_up_flag
        self.server_schedule = pd.DataFrame(server_schedule, columns=range(total_hours), index=range(max_servers))
        self.max_servers = len(self.server_schedule)
        self.num_servers = np.sum(self.server_schedule, axis=0) # number of available servers each hour
        self.server_rates = server_rates # average server_rates per hour (24 array)
        self.service_mean_types = service_mean_types # average service times for each type of customer (array 3*1)

        # server status 
        # Status = 0 (unavailable), 1 (idle), 2 (busy)
        self.server_status = pd.DataFrame(columns=['Status', 'Arr_num', 'Start', 'Finish', 'Wait', 'Type'], index=range(self.max_servers))
        self.hour = ((np.sum(self.server_schedule, axis=0)) > 0).idxmax() #when servers/customers first arrive
        # server status initialization in that hour
        self.available = self.num_servers[self.hour]
        self.server_status['Status'] = [0]*self.max_servers
        self.server_status.loc[range(self.available), 'Status'] = [1] * self.available # w.l.o.g. lower index servers are available
        self.server_status.loc[:, 'Start'] = [-float('inf')] * self.max_servers
        self.server_status.loc[:, 'Finish'] = [float('inf')] * self.max_servers
        # initializing queue and numbers in queue (one for each type)
        self.queue = []
        self.in_queue = [0]*self.cust_types
        # prepapring "self.summary" output df, and initializing it
        cols = ['Arrival_num', 'Arrival_time', 'Checkin_time', 'Service_time', 'Exit_time', 'type', 'n_servers',
                'n_available', 'n_not_available', 'LSD', 'mu', 'abandonments', 'teta', 'previous_WT',
                'WT_QTD', 'WT_QTD_bis', 'LES', 'HOL', 'Real_WT']
        for i in range(self.cust_types):
            cols.extend(['queue' + str(i+1)])
        self.summary = pd.DataFrame(columns=cols, index=range(self.num_arrivals))
        self.summary.loc[:, 'Arrival_num'] = self.arrivals['Arrival_num']
        self.summary.loc[:, 'Arrival_time'] = self.arrivals['time']
        self.summary.loc[:, 'Exit_time'] = float('inf')*self.num_arrivals
        # initializing lists of times of arrivals, abandons, departures
        self.list_t_arrival = list(self.arrivals['time'])
        self.list_t_abandon = list(self.arrivals['time']+self.arrivals['patience'])
        self.list_t_depart = list(self.server_status['Finish'])
        # initializing the clock
        self.clock = 0.0 + self.hour
        self.max_time = float(total_hours)
        #print(cols)
        #print(self.summary)
   
    def run(self):
        while (min(self.list_t_arrival) < self.max_time) or (min(self.list_t_depart) < self.max_time) or not np.all((np.array(self.in_queue) == 0)):
        #while (min(self.list_t_arrival) < self.max_time) or (min(self.list_t_depart) < 50):
            self.advance_time()

            #print(self.clock)

        # Global variables - correct for all the lines in summary
        self.summary['Week'] = [self.week] * self.num_arrivals
        self.summary['Day'] = [self.day] * self.num_arrivals
        #self.summary['Real_WT'] = self.summary['Checkin_time'] - self.summary['Arrival_time']

    def advance_time(self):

        self.t_next_hour = self.hour + 1
        self.t_depart = min(self.list_t_depart)

        self.t_abandon = min(self.list_t_abandon)
        self.t_arrival = min(self.list_t_arrival)
        
        t_event = min(self.t_arrival, self.t_depart, self.t_abandon, self.t_next_hour)
        self.clock = t_event

        # if self.clock == float('inf'):
        #     break
        if self.t_depart == t_event:
            self.handle_depart_event()
        elif self.t_arrival == t_event:
            self.handle_arrival_event()
        elif self.t_next_hour == t_event:
            self.handle_change_hour_event()
        elif self.t_abandon == t_event:
            self.handle_abandon_event()

    def handle_arrival_event(self):
        self.arrival_num = np.argmin(self.list_t_arrival)
        if self.arrival_num > 10:
             print('Breakpoint')
        #print(self.arrival_num)
        self.list_t_arrival[self.arrival_num] = float('inf')
        self.summary.loc[self.arrival_num, 'Arrival_time'] = self.clock
        self.summary.loc[self.arrival_num, 'n_available'] = len(self.server_status.loc[self.server_status.Status == 1])
        self.summary.loc[self.arrival_num, 'n_not_available'] = len(self.server_status.loc[self.server_status.Status == 2])
        self.summary.loc[self.arrival_num, 'n_servers'] = num_servers[self.hour]
        self.summary.loc[self.arrival_num, 'type'] = self.arrivals['type'].iloc[self.arrival_num]

        if self.arrival_num >= 1:
            window = self.summary.loc[self.arrival_num - 1]['queue1'] + self.summary.loc[self.arrival_num - 1][
                'queue2'] + 100  # the rolling's window take into account the 100 previous customers that got a service
            self.summary.loc[self.arrival_num, 'mu'] = 1/self.summary['Service_time'].rolling(window=window, min_periods=1).mean().iloc[self.arrival_num]
            self.summary.loc[self.arrival_num, 'abandonments'] = (self.summary['Checkin_time'] == np.inf).rolling(window=window, min_periods=1).sum().iloc[self.arrival_num]
            self.summary.loc[self.arrival_num, 'previous_WT'] = self.summary['Real_WT'].rolling(window=window, min_periods=1).sum().iloc[self.arrival_num]
        else: # For the first arrival event of the day each parameter is update manually
            self.summary.loc[self.arrival_num, 'mu'] = 1/self.service_mean_types[self.summary.loc[self.arrival_num, 'type']][self.hour].item()
            self.summary.loc[self.arrival_num, 'abandonments'] = 0
            self.summary.loc[self.arrival_num, 'previous_WT'] = 0

        if self.summary.loc[self.arrival_num, 'previous_WT'] == 0: #teta definition
            self.summary.loc[self.arrival_num, 'teta'] = 0
        else:
            self.summary.loc[self.arrival_num, 'teta'] = self.summary.loc[self.arrival_num, 'abandonments']/\
                                                         self.summary.loc[self.arrival_num, 'previous_WT']
        #
        inds = np.array((self.server_status.iloc[:, 0] == 1)) & np.array((self.server_schedule.iloc[:, self.hour] > 0))  #test for idle
        # print(np.sum(inds)," -arrival")
        if np.sum(inds) == 0: # no available server and therefore enters queue
            #generate HOL, LES, in_queue(by types) and put in summary
            if len(self.queue) > 0:
                hol_num = self.queue[0][0]
                self.summary.loc[self.arrival_num, 'HOL'] = self.clock - self.arrivals.time[hol_num]
            else:
                self.summary.loc[self.arrival_num, 'HOL'] = 0

            les_id = np.argmax(np.array(self.server_status['Start']))
            #les_id = np.argmax(np.array(self.server_status['Start'].loc[np.where(self.server_status['Start'] > 0)[0]]))
            self.summary.loc[self.arrival_num, 'LES'] = self.server_status.loc[les_id, 'Wait']
            self.summary.iloc[self.arrival_num, - self.cust_types:] = self.in_queue
            self.summary.loc[self.arrival_num, 'WT_QTD'] = \
                (len(self.queue) + 1) / (self.num_servers[self.hour] * self.server_rates[0][self.hour])
            self.summary.loc[self.arrival_num, 'WT_QTD_bis'] = np.sum(1/(self.summary.loc[self.arrival_num, 'n_servers']*self.summary.loc[self.arrival_num, 'mu'] +
                                                                         self.summary.loc[self.arrival_num, 'teta']*np.arange(self.summary.loc[self.arrival_num, 'queue1']+
                                                                                                                              self.summary.loc[self.arrival_num, 'queue2']+1))) #TODO: check the formula
            if np.where(((self.summary.Service_time.notna()) & ((self.summary.Exit_time < self.clock))))[0].size != 0:
                lsd_idx = self.summary.index[
                        self.summary['Exit_time'] == max(self.summary.iloc[np.where(((self.summary.Service_time.notna()) & ((self.summary.Exit_time < self.clock))))[0]]['Exit_time'])]
                self.summary.loc[self.arrival_num, ['LSD']] = float(self.summary.loc[lsd_idx[-1], 'Service_time'])
            else:
                self.summary.loc[self.arrival_num, ['LSD']] = 0
            # update queue information
            self.queue.append([self.arrival_num, self.arrivals.type[self.arrival_num]])
            self.in_queue[self.queue[-1][1]] += 1
        else: # goes straight into service
            server_num = np.argmax(inds)
            self.server_status.loc[server_num, 'Status'] = 2 #becomes busy
            self.server_status.loc[server_num, 'Arr_num'] = self.arrival_num
            self.server_status.loc[server_num, 'Start'] = self.t_arrival
            self.customer_type = self.arrivals.loc[self.arrival_num, 'type']
            service_time = self.generate_service() # use type and possible other state variables
            self.server_status.loc[server_num, 'Finish'] = self.t_arrival+service_time
            self.server_status.loc[server_num, 'Wait'] = 0
            self.server_status.loc[server_num, 'Type'] = self.customer_type
            self.summary.loc[self.arrival_num, ['Checkin_time', 'Service_time', 'Exit_time', 'Real_WT']] = \
                      [self.clock, service_time, self.clock+service_time, 0]
            self.summary.loc[self.arrival_num, ['LSD', 'WT_QTD', 'WT_QTD_bis', 'LES', 'HOL']] = [None, 0., 0., 0., 0.]
            self.summary.iloc[self.arrival_num, -self.cust_types:] = [0, 0]
            if np.where(((self.summary.Service_time.notna()) & ((self.summary.Exit_time < self.clock))))[0].size != 0:
                lsd_idx = self.summary.index[
                        self.summary['Exit_time'] == max(self.summary.iloc[np.where(((self.summary.Service_time.notna()) \
                                                                                     & ((self.summary.Exit_time < self.clock))))[0]]['Exit_time'])]
                self.summary.loc[self.arrival_num, ['LSD']] = float(self.summary.loc[lsd_idx[-1], 'Service_time'])

            else:
                self.summary.loc[self.arrival_num, ['LSD']] = 0
            self.list_t_abandon[self.arrival_num] = float('inf')
            self.list_t_depart = list(self.server_status['Finish'])
  
    ##
    def handle_depart_event(self):
        server_idx = np.argmin(list(self.server_status['Finish']))
        #print(self.max_servers,' max_server')
        #print(server_idx, ' server_idx')
        self.server_status.iloc[server_idx, :] = [1, None, None, float('inf'), None, None]

        if self.server_schedule.iloc[server_idx, self.hour] == 0:
            #self.server_status.loc[server_idx, 'Status'] = 0
            self.server_status.loc[server_idx, :] = [0, np.nan, float('-inf'), float('inf'), np.nan, np.nan]
        
        if (len(self.queue) > 0) & (self.server_status.loc[server_idx, 'Status'] == 1): #assuming for now no priorities
            customer = self.queue[0]
            customer_id = customer[0]
            self.customer_type = customer[1]
            self.server_status.loc[server_idx, ['Status', 'Arr_num', 'Start', 'Wait', 'Type']] = \
                [2, customer_id, self.clock, self.clock - self.arrivals.loc[customer_id, 'time'], self.customer_type]
            service_time = self.generate_service() # use type and possible other state variables
            self.server_status.loc[server_idx, 'Finish']=self.clock + service_time
            self.summary.loc[customer_id, 'Checkin_time'] = self.clock # complete checkin time
            self.summary.loc[customer_id, 'Exit_time'] = self.clock + service_time  # complete exit time
            self.summary.loc[customer_id, 'Service_time'] = service_time  # complete exit time
            self.summary.loc[customer_id, 'Real_WT'] = self.clock - self.summary.loc[customer_id, 'Arrival_time']
            self.list_t_abandon[customer_id] = float('inf')
            self.in_queue[self.customer_type] -= 1
            self.queue.pop(0)
        self.list_t_depart[server_idx] = self.server_status.Finish[server_idx]

    def handle_change_hour_event(self):
        self.hour += 1
        for indx in range(self.max_servers):
            if (self.server_status.loc[indx, 'Status'] == 1) & (self.server_schedule.iloc[indx, self.hour] == 0):
                self.server_status.iloc[indx, :] = [0, None, None, float('inf'), None, None]
            if (self.server_status.loc[indx, 'Status'] == 0) & (self.server_schedule.iloc[indx, self.hour] == 1):
                self.server_status.iloc[indx, :] = [1, None, None, float('inf'), None, None]
        self.list_t_depart = list(self.server_status['Finish'])
        
    def handle_abandon_event(self):
        customer_id = np.argmin(self.list_t_abandon)
        self.list_t_abandon[customer_id] = float('inf')
        self.summary.loc[customer_id, ['Checkin_time', 'Service_time', 'Exit_time', 'Real_WT']] = [float('inf'), None, self.clock, self.clock - self.summary.loc[customer_id, 'Arrival_time']]
        for ind in range(len(self.queue)):
            if self.queue[ind][0] == customer_id:
                queue_idx = ind
                self.in_queue[self.queue[ind][1]] -= 1
        self.queue.pop(queue_idx)
    
    def generate_service(self):
        # based on CV = 1 as for exponential    
        custtype = self.customer_type
        queue_length = np.sum(self.in_queue)
        if self.speed_up_flag:
            speed_up = math.exp(-len(self.queue)*0.02)
        else:
            speed_up = 1
        # for speeding up service due to queue length (decreasing service times)
        # e.g. speed_up = 1/(1+queue_length/100)
        # for lognormal (CV=1 => value of sigma, which => mu for achieving mean=1)

        # for exponential
        # serv1 = np.random.exponential(1)       # exponential service time distribution with mean 1
        # make mean appropriate for type of customer, queue length

        if self.service_distribution == 'LogNormal':
            sigma = np.sqrt(np.log(2))
            mu = -0.5 * sigma * sigma
            serv1 = np.random.lognormal(mu, sigma)  # lognormal service time distribution with mean 1
            service = serv1*self.service_mean_types[custtype][self.hour].item()*speed_up
        elif self.service_distribution == 'Exponential':
            service = np.random.exponential(self.service_mean_types[custtype][self.hour].item()*speed_up)
        return service


if __name__ == '__main__':

    np.random.seed(42)
    path = input("Izik's computer? : ")
    shrunk = input("Shrunk system? : ")
    for exp in [12]:
        print('Experiment: ' + str(exp))
        if path == '0':
            current_path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_' + str(exp)
        elif path == '1':
            current_path = r'C:\Users\elishevaz\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_' + str(exp)

        start = time.time()
        df = pd.DataFrame()
        ctypes = 2
        ctypes_dict = {0: 'private', 1: 'not_private'}
        c_std_dict = {0: 'std_private', 1: 'std_not_private'}
        speed_up_flag = 0
        total_hours = 12
        if exp == 0 or exp == 12:
            total_hours = 24
        if exp == 2:
            service_distribution = 'Exponential'
        elif exp != 2:
            service_distribution = 'LogNormal'
        elif exp == 6:
            speed_up_flag = 1

        arr_rates_types = np.zeros((2, total_hours), dtype=int)
        df_arrivals = pd.read_csv(current_path + '/Arrivals.csv')
        df_service_time = pd.read_csv(current_path + '/Service_time.csv')
        df_abandonment = pd.read_csv(current_path + '/Abandonment.csv')
        df_number_of_agents = pd.read_csv(current_path + '/number_of_agents.csv')
        if shrunk == '1':
            divisor = 30
            df_arrivals['private'] = np.ceil(df_arrivals['private'] / divisor).astype(int)
            df_arrivals['not_private'] = np.ceil(df_arrivals['not_private'] / divisor).astype(int)
            df_number_of_agents['number_of_agents'] = np.ceil(df_number_of_agents['number_of_agents']/divisor).astype(int)
            number_of_weeks = np.arange(0, 30, 1)
        else:
            number_of_weeks = np.arange(0, 6, 1)

        for week in number_of_weeks:
            print('---------------------------------Week: ' + str(week) + '---------------------------------')
            for day in [0, 1, 2, 3, 6]: #wihtout saturday
            #for day in [1]:
                print('---------------------------------Day: ' + str(day) + '---------------------------------')
                for type in range(ctypes):
                    if exp == 0:
                        mean = df_arrivals[ctypes_dict[type]].loc[df_arrivals['Weekday'] == day].to_numpy()
                        std = df_arrivals[c_std_dict[type]].loc[df_arrivals['Weekday'] == day].to_numpy()
                        arr_rates_types[type] = [int(np.random.uniform(low= max(0, mean[i]-std[i]), high=(mean[i]+std[i]))) for i in range(len(mean))]
                    else:
                        arr_rates_types[type] = df_arrivals[ctypes_dict[type]].loc[df_arrivals['Weekday'] == day].to_numpy()
                num_servers = df_number_of_agents['number_of_agents'].loc[df_number_of_agents['Weekday'] == day].to_numpy()
                max_servers = np.max(num_servers)
                hours = [num_servers > 0]
                #service_time definition

                if exp == 0: # parameters with std
                    private_mean_s = df_service_time['private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    private_std_s = df_service_time['std_private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    service_time_sample_private = np.array([int(np.random.uniform(low=max(0, private_mean_s[i]-private_std_s[i]),
                                                                              high=(private_mean_s[i]+private_std_s[i]))) for i in range(len(private_mean_s))])

                    not_private_mean_s = df_service_time['not_private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    not_private_std_s = df_service_time['std_not_private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    service_time_sample_not_private = np.array([int(np.random.uniform(low=max(0, not_private_mean_s[i] - not_private_std_s[i]),
                                                                              high=(not_private_mean_s[i] + not_private_std_s[i]))) for i in range(len(not_private_mean_s))])
                else: # parameters without std (mean definition)
                    service_time_sample_not_private = df_service_time['not_private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    service_time_sample_private = df_service_time['private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)

                service_rates = np.divide(np.full((1, total_hours), 3600, dtype=float), service_time_sample_private) #TODO: CHECK IF WE CAN ADD THE NOT PRIVATE SERVICE RATE

                server_schedule = np.reshape([0] * max_servers * total_hours, (max_servers, total_hours))
                for j in range(total_hours):
                    for i in range(max_servers):
                        if i < num_servers[j]:
                            server_schedule[i][j] = 1
                # now call arrivals generator - use arrays "arr_rates_types", "mean_patience_types"
                mean_patience = df_abandonment['wait_time'].loc[df_abandonment['Weekday'] == day].to_numpy()/3600
                #mean_patience_types = np.full((1, ctypes), mean_patience)
                mean_patience_types = [mean_patience, np.array([float('inf')] * total_hours)]
                mean_service_private = service_time_sample_private/3600
                mean_service_not_private = service_time_sample_not_private/3600

                service_mean_types = [mean_service_private, mean_service_not_private]
                #service_std_types = [sigma_private, sigma_not_private]
                arrivals = gen_arrivals(arr_rates_types, mean_patience_types, total_hours)
                # -> produces "arrivals" dataframe with columns: "arr_num","time","patience", "type"
                # then call simulator - use arrays: "arrivals", "server_schedule", "service_mean_types", "service_rates"
                s = Simulator(arrivals, week, day, server_schedule, service_mean_types, service_rates, total_hours=total_hours, service_distribution=service_distribution, speed_up_flag=speed_up_flag)
                s.run()
                df = pd.concat([df, s.summary], ignore_index=True)


        process_time = time.time() - start
        print('Time elapsed: ', int((process_time/60)))
        print(process_time)
        print(process_time/60)
        print('Data Shape: ', df.shape)

        if shrunk == '1':
            df.to_csv(current_path + '/New_features_simulation_lowest_system.csv')
        else:
            df.to_csv(current_path+'/New_features_simulation.csv')


