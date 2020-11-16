import numpy as np

class DataLoader():

    def __init__(self, num, data, week_split=4, abandonment_removed_train=0, abandonment_removed_test=0,  features_list=[], features_to_check=[], beginning=0.75):
        self.num = num
        self.week_split = week_split #6 for 9 weeks simulation, 4 for 6 weeks simulation, 24 for 30 weeks simulation split, inverse split = 2 + change the sign of the test and train
        self.abandonment_removed_train = abandonment_removed_train
        self.abandonment_removed_test = abandonment_removed_test
        self.feature_to_check = features_to_check
        # if self.num in [1, 2, 4, 5, 6, 7, 8]:  # experiment without abandonment
        #     self.features = np.delete(np.array(features_list), 'abandonment') # to avoid NaN after normalizing
        # else:
        self.features = features_list
        self.beginning = beginning
        self.data = data.loc[data.Arrival_time >= beginning]
        if self.num == 0 or self.num == 12:  # removing the night's hours
            self.data = data.loc[(data.Arrival_time >= 8.75) & (data.Arrival_time < 20)]

        self.data_train = self.data.loc[self.data.Week < self.week_split]
        if self.abandonment_removed_train: # in case of training only on those who got served
            self.data_train = self.data_train.replace([np.inf, -np.inf], np.nan).dropna(subset=['Checkin_time'], axis=0, how="any")

        self.data_test = self.data.loc[self.data.Week >= self.week_split]
        if self.abandonment_removed_test:  # in case of training only on those who got served
            self.data_test = self.data_test.replace([np.inf, -np.inf], np.nan).dropna(subset=['Checkin_time'], axis=0, how="any")



    #data.loc[data.Real_WT == np.inf, 'Real_WT'] = data.loc[data.Real_WT == np.inf, 'Exit_time'] \- data.loc[data.Real_WT == np.inf, 'Arrival_time'] #TODO: check if necessary
    #data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Real_WT'], axis=0, how="any")#remove all the missing value #TODO: check if necessary

    def print_summary(self):
        print('----------------------------------------------SUMMARY-------------------------------------------------------------')
        print('-------------------------------------- Experiment ' + str(self.num)+ ' loading --------------------------------------')
        print('Features set: ', self.features)
        if self.abandonment_removed_train:
            print('The training is based only on customers that got served')
        else:
            print('The training is based on the whole dataset')
        if self.abandonment_removed_test:
            print('The data is tested only on customers that got served')
        else:
            print('The data is tested on the whole data')
        if ((self.data_train[self.features].isna().sum() != 0).any()) or ((self.data_test[self.features].isna().sum() != 0).any()):
            print('MISSING VALUES !!!!!!!')

        print('--------------------------------------------------------------------------------------------------------------------')


    def extract_train(self):
        print('------------------------------------------------Training set extraction----------------------------------------------------')

        #split features/label
        self.X_train = self.data_train[self.features]
        self.X_checked_train = self.data_train[self.feature_to_check]
        self.y_train = self.data_train['Real_WT'].to_frame()*3600

        #reindexing
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_checked_train.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)

        return self.X_train, self.X_checked_train, self.y_train.to_numpy().reshape(-1), float(self.y_train.mean())

    def extract_test(self):
        print('------------------------------------------------Test set extraction----------------------------------------------------')

        #split features/label
        self.X_test = self.data_test[self.features]
        self.X_checked_test = self.data_test[self.feature_to_check]
        self.y_test = self.data_test['Real_WT'].to_frame()*3600

        #reindexing
        self.X_test.reset_index(drop=True, inplace=True)
        self.X_checked_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)
        return self.X_test, self.X_checked_test, self.y_test.to_numpy().reshape(-1), float(self.y_test.mean())

    def extract_QT_predictors_train(self):
        print(
            '------------------------------------------------QT_predictors train extraction----------------------------------------------------')
        self.y_QTP_train = self.data_train['QTP_data_based'].to_frame() * 3600
        self.y_QTP_weighted_train = self.data_train['QTP_weighted'].to_frame() * 3600
        self.y_QTP_train.reset_index(drop=True, inplace=True)
        self.y_QTP_weighted_train.reset_index(drop=True, inplace=True)

        return self.y_QTP_train.to_numpy().reshape(-1), self.y_QTP_weighted_train.to_numpy().reshape(-1)

    def extract_QT_predictors_test(self):
        print('------------------------------------------------QT_predictors test extraction----------------------------------------------------')
        self.y_QTP_test = self.data_test['QTP_data_based'].to_frame()*3600
        self.y_QTP_weighted_test = self.data_test['QTP_weighted'].to_frame()*3600
        self.y_QTP_weighted_bis_test = self.data_test['QTP_weighted_bis'].to_frame() * 3600

        self.y_QTP_test.reset_index(drop=True, inplace=True)
        self.y_QTP_weighted_test.reset_index(drop=True, inplace=True)
        self.y_QTP_weighted_bis_test.reset_index(drop=True, inplace=True)

        return self.y_QTP_test.to_numpy().reshape(-1), self.y_QTP_weighted_test.to_numpy().reshape(-1), self.y_QTP_weighted_bis_test.to_numpy().reshape(-1)

    def extract_snapshot_train(self):
        print('------------------------------------------------Snapshot train extraction----------------------------------------------------')
        self.y_HOL_train = self.data_train['HOL'].to_frame()*3600
        self.y_LES_train = self.data_train['LES'].to_frame()*3600
        self.y_HOL_train.reset_index(drop=True, inplace=True)
        self.y_LES_train.reset_index(drop=True, inplace=True)

        return self.y_HOL_train.to_numpy().reshape(-1), self.y_LES_train.to_numpy().reshape(-1)

    def extract_snapshot_test(self):
        print('------------------------------------------------Snapshot test extraction----------------------------------------------------')
        self.y_HOL_test = self.data_test['HOL'].to_frame()*3600
        self.y_LES_test = self.data_test['LES'].to_frame()*3600
        self.y_HOL_test.reset_index(drop=True, inplace=True)
        self.y_LES_test.reset_index(drop=True, inplace=True)

        return self.y_HOL_test.to_numpy().reshape(-1), self.y_LES_test.to_numpy().reshape(-1)


    def extract_no_zero(self):
        print('------------------------------------------------No zero predictors extraction----------------------------------------------------')
        self.y_test_not_zero = np.array(self.y_test.loc[self.y_test.Real_WT > 2]).reshape(-1)  # y that are not zero
        self.y_QTP_test_not_zero = self.y_QTP.to_numpy()[self.y_test.loc[self.y_test.Real_WT > 2].index].reshape(-1)
        self.y_QTP_weighted_test_not_zero = self.y_QTP_weighted.to_numpy()[self.y_test.loc[self.y_test.Real_WT > 2 ].index].reshape(-1)
        self.y_LES_test_not_zero = self.y_LES.to_numpy()[self.y_test.loc[self.y_test.Real_WT > 2].index].reshape(-1)
        self.y_HOL_test_not_zero = self.y_HOL.to_numpy()[self.y_test.loc[self.y_test.Real_WT > 2].index].reshape(-1)

        return self.y_test_not_zero, self.y_QTP_test_not_zero, self.y_QTP_weighted_test_not_zero, self.y_LES_test_not_zero, self.y_HOL_test_not_zero


