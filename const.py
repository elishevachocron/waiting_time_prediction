# ------------------------------------------------------------------- #
# ---------------------- MODELS HYPERPARAMETERS --------------------- #
# ------------------------------------------------------------------- #
algorithms_used = ['RF', 'SVR', 'GBR', 'ANN', 'QTD', 'LES', 'HOL']
n_folds = 5
model_state = {'RF': 'saved', 'SVR': 'saved', 'GBR': 'saved', 'ANN': 'saved'} #"fit" the model, not using the "saved" model

# ---------------------- Random Forest ------------------------------ #
grid_values_rf = {'max_depth': [1, 3, 6, 10, 15], 'n_estimators': [70, 100, 150, 200], 'n_jobs': [10]}

# ---------------------------- SVR ---------------------------------- #
grid_values_svr = {'kernel': ['linear', 'rbf'], 'C': [0.001, 0.01, 1],\
                   'max_iter': [4000], 'epsilon': [0.001, 0.009, 0.01], 'gamma': ['scale']}

# ---------------------------- GBR ---------------------------------- #
grid_values_gbr = {'loss': ['ls', 'lad'], 'learning_rate': [0.01, 0.05, 0.1],
                   'n_estimators': [50, 150, 200], 'max_depth': [3, 10, 15],
                   'max_leaf_nodes': [30], 'validation_fraction': [0.2],
                   'n_iter_no_change': [10], 'tol': [0.0001]}

# ----------------------------- ANN --------------------------------- #
grid_values_ann = {'hidden_layer_sizes': [5, 30, 80],\
                 'activation': ['identity', 'logistic', 'relu'], 'learning_rate': ['invscaling', 'adaptive']}

# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #

features_type = {'set_features_1': ['Arrival_time', 'Day', 'n_available', 'n_not_available', 'LSD', 'queue1', 'queue2', 'mu', 'abandonments', 'previous_WT'],
                 'set_features_2': ['Arrival_time', 'Day', 'n_available', 'n_not_available', 'LSD', 'queue1', 'queue2', 'mu', 'abandonments', 'previous_WT', 'HOL', 'LES']}

columns_list = ['Arrival_time', 'Day', 'n_available', 'n_not_available', 'LSD', 'queue1', 'queue2', 'mu', 'abandonments', 'previous_WT', 'WT_QTD', 'LES', 'HOL', 'Real_WT']

computer = input("Izik's Computer?: ")

if computer == '0':
    path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'
else:
    path = r'C:\Users\elishevaz\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_'