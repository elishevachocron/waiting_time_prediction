import numpy as np
import pandas as pd
import math
import time

start = time.time()

def arrival_rate_func(experiment):
    if experiment == 0:
        ch_params = {0: (17, 68)}
    elif experiment == 1:
        ch_params = {0: (17, 77)}
    elif experiment == 2:
        ch_params = {0: (17, 81)}
    elif experiment == 3:
        ch_params = {0: (16, 56)}
    elif experiment == 4:
        ch_params = {0: (16, 51)}
    elif experiment == 5:
        ch_params = {0: (16, 64)}
    elif experiment == 6:
        ch_params = {0: (17, 61)}
    elif experiment == 7:
        ch_params = {0: (17, 77)}
    elif experiment == 8:
        ch_params = {0: (17, 65)}
    elif experiment == 9:
        ch_params = {0: (17, 81)}

    return ch_params


class Simulator():

    def __init__(self, max_time=16.0, day=1, shift_dicts={0: [[0, 16]]}, ch_params={0: (17, 80)}):
        self.s_rate = ch_params[0][0]
        self.a_rate = ch_params[0][1]
        self.clock = 0.0
        self.ch_params = ch_params

        self.in_service = 0
        self.WT_QTD = 0
        self.HOL_idx = 0
        self.LES_idx = 0
        self.LSD_idx = 0 #last service duration
        self.in_queue = 0
        self.max_time = max_time
        self.customers_in_queue = []
        self.day = day
        self.masks_dict = self.create_masks(shift_dicts)
        self.active_servers = self.masks_dict[self.clock]
        self.n_servers = np.sum(self.active_servers)
        self.list_t_depart = [float('inf')] * len(self.active_servers)
        self.customers_in_service = [None] * len(self.active_servers)

        self.summary = pd.DataFrame(columns=['Arrival_time', 'Checkin_time', 'Service_time', 'Exit_time','Length_of_queue',
                                             's_rate', 'a_rate', 'n_servers', 'LSD', 'Utility', 'p0', 'WT_QTT', 'WT_QTD', 'LES', 'HOL'])

        self.t_arrival = self.generate_interarrival()
        self.t_depart = float('inf')

        self.num_arrivals = 0
        self.num_departs = 0
        self.compute_qt_params()

    def run(self):
        while (self.clock <= self.max_time) or (self.in_service + self.in_queue > 0):
            # if self.in_queue > 0:
            #     print('Hello')
            self.advance_time()
            #print(self.clock)

        # Global variables - correct for all the lines in summary
        self.summary['Day'] = [self.day] * self.num_arrivals
        self.summary['Real_WT'] = self.summary['Checkin_time'] - self.summary['Arrival_time']

    def advance_time(self):

        if len(self.masks_dict) <= 1: #check if a change in shift occurs
            next_change_shift_time = float('inf')
        else:
            next_change_shift_time = min(list(self.masks_dict.keys()))

        if len(self.ch_params) <= 1:#check if a change in service or arrival rate occurs
            next_change_params_time = float('inf')
        else:
            next_change_params_time = min(list(self.ch_params.keys()))

        self.t_depart = min(self.list_t_depart)
        t_event = min(self.t_arrival, self.t_depart, next_change_params_time, next_change_shift_time)
        self.clock = t_event
        # if self.clock == float('inf'):
        #     break
        if self.t_depart == t_event:
            self.handle_depart_event()
        elif self.t_arrival == t_event:
            self.handle_arrival_event()
        elif next_change_params_time == t_event:
            self.handle_change_params_event()
        elif next_change_shift_time == t_event:
            self.handle_change_shift_event()

    def handle_arrival_event(self):

        self.summary.loc[self.num_arrivals] = [self.t_arrival, None, None, None, self.in_queue,
                                               self.s_rate, self.a_rate, self.n_servers, None, self.rho, self.p0,
                                               self.theorical_Lq/self.a_rate, self.WT_QTD, None, None]
        curr_service_time = self.generate_service()
        num_in_system = self.in_queue + self.in_service + 1  # 1 for the new customer

        #Snapshot Predictors index definition
        if (any(i != None for i in self.customers_in_service)):# check if someone is in service
            self.LES_idx = max([i for i in self.customers_in_service if type(i) == int]) #take the max (the index of the last customer), if the condition is not respected, LES_idx = last update

        # Last Service Duration
        if any(self.summary['Exit_time']< self.summary.loc[self.num_arrivals, 'Arrival_time']):
            self.LSD_idx = \
                max(self.summary.index[self.summary['Exit_time']< self.summary.loc[self.num_arrivals, 'Arrival_time']].tolist())
                # max (last) of list of customers that finished their service before the arrival of the new customer

        #self.summary.loc[self.num_arrivals, 'LES'] = self.clock - self.summary.loc[self.HOL_idx]['Arrival_time']

        if len(self.customers_in_queue) > 0:
            self.HOL_idx = self.customers_in_queue[0]
            self.summary.loc[self.num_arrivals, 'HOL'] = self.clock - self.summary.loc[self.HOL_idx]['Arrival_time']
        else:
            self.summary.loc[self.num_arrivals, 'HOL'] = 0

        if num_in_system <= self.n_servers:
            self.WT_QTD = 0
            available_servers = np.where(np.logical_and(np.array(self.active_servers), np.array(self.customers_in_service) == None))[0]
            self.customers_in_service[available_servers[0]] = self.num_arrivals
            self.list_t_depart[available_servers[0]] = self.clock + curr_service_time
            self.summary.loc[self.num_arrivals, ['Checkin_time', 'Service_time', 'Exit_time']] = [self.t_arrival, curr_service_time, self.clock+curr_service_time]
            self.summary.loc[self.num_arrivals, 'LES'] = self.summary.loc[self.LES_idx]['Checkin_time'] - \
                                                         self.summary.loc[self.LES_idx][
                                                             'Arrival_time']  # take the WT of the LES_idx
            self.in_service += 1
        else:
            self.WT_QTD = (self.in_queue + 1) / (self.n_servers * self.s_rate)
            self.summary.loc[self.num_arrivals, ['Checkin_time', 'Service_time', 'Exit_time', 'LES']] = [None, curr_service_time, None, \
                                                                                                         self.summary.loc[self.LES_idx]['Checkin_time'] \
                                                                                                         - self.summary.loc[self.LES_idx]['Arrival_time']]
            self.customers_in_queue += [self.num_arrivals]
            self.in_queue += 1

        if self.LSD_idx:
            self.summary.loc[self.num_arrivals, 'LSD'] = float(self.summary.loc[self.LSD_idx, ['Service_time']]) # take the service time of the last_service_duration_idx
        else:
            self.summary.loc[self.num_arrivals, 'LSD'] = 0

        if self.clock <= self.max_time:
            self.t_arrival = self.clock + self.generate_interarrival()
            if self.t_arrival > self.max_time:  # Boundary case: Generated arrival is greater than limit
                self.t_arrival = float('inf')
            self.num_arrivals += 1
        else:
            self.t_arrival = float('inf')

    def handle_depart_event(self):
        self.num_departs += 1
        min_idx = np.argmin(self.list_t_depart)
        self.list_t_depart[min_idx] = float('inf')
        self.customers_in_service[min_idx] = None
        self.in_service -= 1
        if self.customers_in_queue and self.active_servers[min_idx]:
            curr_customer = self.customers_in_queue[0]
            self.customers_in_queue = self.customers_in_queue[1:]  # to pop the current customer from the queue
            self.summary.loc[curr_customer, 'Checkin_time'] = self.clock # complete checkin time
            self.summary.loc[curr_customer]['Exit_time'] = self.clock + self.summary.loc[curr_customer]['Service_time']  # complete exit time
            self.customers_in_service[min_idx] = curr_customer
            self.list_t_depart[min_idx] = self.summary.loc[curr_customer, 'Exit_time']
            self.in_service += 1
            self.in_queue -= 1

    def handle_change_params_event(self):
        self.s_rate = self.ch_params[self.clock][0]
        self.a_rate = self.ch_params[self.clock][1]
        self.compute_qt_params()
        del self.ch_params[self.clock]

    def handle_change_shift_event(self):
        self.active_servers = self.masks_dict[self.clock]
        self.n_servers = np.sum(self.active_servers)
        self.compute_qt_params()
        del self.masks_dict[self.clock]

    def create_masks(self, shifts_dict):

        masks_dict = {}
        for server, shifts in shifts_dict.items():
            for event in shifts:
                masks_dict[event[0]] = [False] * len(shifts_dict)
                masks_dict[event[1]] = [False] * len(shifts_dict)

        for server, shifts in shifts_dict.items():
            for event in shifts:
                all_keys = np.array(list(masks_dict.keys()))
                keys_to_modify = np.logical_and(all_keys >= event[0], all_keys < event[1])
                keys_to_modify = all_keys[keys_to_modify]
                for key in keys_to_modify:
                    masks_dict[key][server] = True
        # Removing last key to solve boundary issue
        max_key = max(list(masks_dict.keys()))
        del masks_dict[max_key]
        return masks_dict

    def compute_qt_params(self):
        self.rho = self.a_rate/(self.n_servers*self.s_rate) #utility service
        self.p0 = 1 / np.sum([((math.pow((self.n_servers * self.rho), idx) / math.factorial(idx)) + (
                    math.pow((self.n_servers * self.rho), self.n_servers) / (math.factorial(self.n_servers) * (1 - self.rho)))) for idx in
                         range(self.n_servers)])
        self.theorical_Lq = (self.p0*math.pow((self.a_rate/self.s_rate), self.n_servers)*self.rho)/(math.factorial(self.n_servers)*math.pow((1-self.rho), 2))


    def generate_interarrival(self):
        return np.random.exponential(1. / self.a_rate)

    def generate_service(self):
        return np.random.lognormal(mean=4.65, sigma=1.2)/3600 #lognormal service time distribution in hour
        #return np.random.exponential(1. / self.s_rate) #exponential service time distribution


num_row = []
for experiment in np.arange(0, 3, 1):
    df = pd.DataFrame()
    for day in range(120):
        if experiment % 2 == 0:
            shifts = {0: [[0, 16]], 1: [[0, 16]], 2: [[0, 16]], 3: [[0, 16]]}
        else:
            shifts = {0: [[0, 16]], 1: [[0, 16]], 2: [[0, 16]], 3: [[0, 16]], 4: [[0, 16]]}
        ch_params = arrival_rate_func(experiment)
        np.random.seed(day)
        shifts = {0: [[0, 16]], 1: [[0, 16]], 2: [[0, 16]], 3: [[0, 16]], 4: [[0, 16]]} #for lognormal service time
        print('---------------------- Day: ' + str(day) + ', Experiment: ' + str(experiment+1) + ' ---------------------- ')
        s = Simulator(day=day, shift_dicts=shifts, ch_params=ch_params)
        s.run()
        df = pd.concat([df, s.summary], ignore_index=True)

    num_row.append(len(df))
    df.to_csv(r'C:\Users\Elisheva\Dropbox\QueueMining\WorkingDocs\Simulator\Queue_Lognormal_service_time\Experiments\Exp_' + str(experiment+1)\
              + '\Experiment_' + str(experiment+1) + '.csv')


end = time.time()

print('------------------Minutes: ' + str(int( (end-time)/60) )+ ' ------------------')
# np.save(r'C:\Users\Elisheva\Dropbox\QueueMining\WorkingDocs\Simulator\Perfect_Queue\num_rows', num_row)
# print(num_row)

#print(df)

#[(math.pow((s.n_servers*rho), idx)/math.factorial(idx))+(math.pow((s.n_servers*rho), s.n_servers)/(math.factorial(s.n_servers)*(1-rho)))]