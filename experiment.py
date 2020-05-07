from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
path_to_save = 'C:\\Users\\Elisheva\\Dropbox\\QueueMining\\WorkingDocs\\Simulator\\Check Plot'
path = r'C:\Users\Elisheva\Dropbox\QueueMining\WorkingDocs\Simulator\Perfect_Queue\Experiments\Exp_9'
X_train = pd.read_csv(path + '\\' + 'experiment_features_type_3_X_train.csv')
X_test = [i for i in np.arange(0, 101, 1)]
y_train = pd.read_csv(path + '\\' + 'experiment_features_type_3_y_train.csv')
# ---------------------- Random Forest ------------------------------- #
grid_values_rf = {'max_depth': [1, 3, 10], 'n_estimators': [10, 50, 150]}
# ---------------------------- SVR ---------------------------------- #
grid_values_svr = {'kernel': ('linear', 'rbf'), 'C': [0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25]}
# ---------------------------- GBR ---------------------------------- #
grid_values_gbr = {'loss': ['ls', 'lad'], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [10, 50, 150]}
# ----------------------------- ANN ----------------------------------- #
grid_values_ann = {'hidden_layer_sizes': [5, 30, 80],\
                 'activation': ['identity', 'logistic', 'relu'], 'learning_rate': ['invscaling', 'adaptive']}
rf_reg = RandomForestRegressor()
grid_reg_mse_rf = GridSearchCV(rf_reg, param_grid=grid_values_rf, scoring='neg_mean_squared_error')
svr_reg = SVR()
grid_reg_mse_svr = GridSearchCV(svr_reg, param_grid=grid_values_svr, scoring='neg_mean_squared_error')
gbr_reg = GradientBoostingRegressor()
grid_reg_mse_gbr = GridSearchCV(gbr_reg, param_grid=grid_values_gbr, scoring='neg_mean_squared_error')
ann_reg = MLPRegressor()
grid_reg_mse_ann = GridSearchCV(ann_reg, param_grid=grid_values_ann, scoring='neg_mean_squared_error')
print('---------------RF FITTING---------------')
grid_reg_mse_rf.fit(X_train, y_train)
print('---------------SVR FITTING---------------')
grid_reg_mse_svr.fit(X_train, y_train)
print('---------------GBR FITTING---------------')
grid_reg_mse_gbr.fit(X_train, y_train)
print('---------------ANN FITTING---------------')
grid_reg_mse_ann.fit(X_train, y_train)
X = pd.DataFrame(X_test)
y_pred_rf = grid_reg_mse_rf.predict(X)
y_pred_svr = grid_reg_mse_svr.predict(X)
y_pred_gbr = grid_reg_mse_gbr.predict(X)
y_pred_ann = grid_reg_mse_ann.predict(X)
plt.figure()
plt.scatter(X, y_pred_rf)
plt.xlabel('QL')
plt.ylabel('WT')
plt.title('Decision Function RF')
plt.savefig(path_to_save + '/' + 'RF_DecisionFunction.png')
plt.figure()
plt.scatter(X, y_pred_svr)
plt.xlabel('QL')
plt.ylabel('WT')
plt.title('Decision Function SVR')
plt.savefig(path_to_save + '/' + 'SVR_DecisionFunction.png')
plt.figure()
plt.scatter(X, y_pred_gbr)
plt.xlabel('QL')
plt.ylabel('WT')
plt.title('Decision Function GBR')
plt.savefig(path_to_save + '/' + 'GBR_DecisionFunction.png')
plt.figure()
plt.scatter(X, y_pred_ann)
plt.xlabel('QL')
plt.ylabel('WT')
plt.title('Decision Function ANN')
plt.savefig(path_to_save + '/' + 'ANN_DecisionFunction.png')

