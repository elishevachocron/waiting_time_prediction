# ------------------------------------------------------------------- #
# ---------------------- MODELS HYPERPARAMETERS --------------------- #
# ------------------------------------------------------------------- #
algorithms_used = ['RF', 'SVR', 'GBR', 'XGBR', 'ANN', 'QTD', 'LES', 'HOL']
n_folds = 5
model_state = {'RF': 'saved', 'SVR': 'saved', 'GBR': 'fit', 'XGBR': 'fit', 'ANN': 'fit'} # "fit" the model, not using the "saved" model

# ---------------------- Random Forest ------------------------------ #
grid_values_rf = {'max_depth': [1, 3, 6, 10, 15], 'n_estimators': [100, 150, 200, 250], 'n_jobs': [15]}

# ---------------------------- SVR ---------------------------------- #
grid_values_svr = {'kernel': ['linear', 'rbf'], 'C': [0.001, 0.009, 0.01, 0.09, 1],\
                   'max_iter': [500, 1000, 3000], 'epsilon': [0.001, 0.009, 0.01]}

# ---------------------------- GBR ---------------------------------- #
grid_values_gbr = {'loss': ['ls', 'lad'], 'learning_rate': [0.01, 0.05, 0.1],
                   'n_estimators': [50, 150, 200], 'max_depth': [3, 10, 15]}
# ---------------------------- XGBR ---------------------------------- #
grid_values_xgbr = {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [10, 50, 150], 'max_depth': [1, 3, 10, 15], 'n_jobs': [15]}
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