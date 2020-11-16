import socket
import numpy as np
import time
import warnings


#######################################################################################################################
#PARAMETERS DEFINITION
exp_beginning = {0: 8.75, 1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75, 9: 0.75, 10: 0.75, 11: 0.75, 12: 8.75, 13: 0.75, 14: 0.75, 15: 0.75}
quantile = 1
delta = 10
train_abandonment_removed = 0
test_abandonment_removed = 0
features = ['Arrival_time', 'Day', 'n_available', 'n_not_available', 'LSD', 'queue1', 'queue2', 'mu', 'abandonment',
            'previous_WT', 'in_service_p', 'in_service_np']
features_to_check = ['Arrival_time', 'Day', 'n_available', 'n_not_available', 'LSD', 'queue1', 'queue2', 'mu',
                     'mu_private', 'mu_not_private', 'mu_weighted', 'abandonment', 'previous_WT', 'in_service_p',
                     'in_service_np', 'QTP_data_based', 'QTP_weighted', 'QTP_weighted_bis']
title = 'general' #TODO: implement the name of the specific model
np.random.seed(42)
start = time.time()
warnings.filterwarnings('ignore')
hostname = socket.gethostname() #to know on which computer we are working
if hostname == 'DESKTOP-A925VLR':
    path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'
else:
    path = r'C:\Users\elishevaz\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'
#######################################################################################################################

# ------------------------------------------------------------------- #
# ---------------------- MODELS HYPERPARAMETERS --------------------- #
# ------------------------------------------------------------------- #
n_folds = 5
algorithms_used = ['LR', 'RF', 'SVR', 'GBR', 'ANN', 'QTP', 'QTP_weighted', 'QTP_weighted_bis', 'LES', 'HOL']
model_state = {'RF': 'fit', 'SVR': 'fit', 'GBR': 'fit', 'ANN': 'fit'} #"fit" the model, not using the "saved" model
# algorithms_used = ['LR', 'RF',  'ANN', 'QTD', 'QTP_new', 'LES', 'HOL']
# model_state = {'RF': 'saved', 'ANN': 'saved'} #"fit" the model, not using the "saved" mode
# ---------------------- Random Forest ------------------------------ #
grid_values_rf = {'max_depth': [1, 3, 6, 10, 15], 'n_estimators': [70, 100, 150, 200], 'n_jobs': [10]}

# ---------------------------- SVR ---------------------------------- #
grid_values_svr = {'kernel': ['linear', 'rbf'] , 'C': [0.001, 0.01, 1],\
                   'max_iter': [4000], 'epsilon': [0.001, 0.009, 0.01] , 'gamma': ['scale']}

# ---------------------------- GBR ---------------------------------- #
grid_values_gbr = {'loss': ['ls', 'lad'],
                   'n_estimators': [50, 70, 100, 150, 200], 'max_depth': [3, 10, 15], 'validation_fraction': [0.2],
                   'n_iter_no_change': [10], 'tol': [0.001], 'max_leaf_nodes': [30]}

# ----------------------------- ANN --------------------------------- #
#'hidden_layer_sizes': [5, 30, 80]
grid_values_ann = {'hidden_layer_sizes': [(5, 1), (30, 1), (80, 1), (80, 30, 1)] ,\
                 'activation': ['relu'], 'learning_rate': ['invscaling', 'adaptive']}

# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #



