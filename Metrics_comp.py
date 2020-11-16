import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

class Evaluation:
    def __init__(self, y, y_hat, quantile):
        self.quantile = quantile
        abs_error = np.absolute(y - y_hat)
        index_quantile = np.where(abs_error < np.quantile(abs_error, self.quantile))[0]
        self.y = y[index_quantile]
        self.y_hat = y_hat[index_quantile]
        self.mean_y = np.mean(self.y)
        self.y_not_zero = self.y[np.where(self.y > 2)[0]]
        self.y_hat_not_zero = self.y_hat[np.where(self.y > 2)[0]]

    def metrics_computation(self):
        self.re = round((np.divide(np.absolute(self.y_not_zero - self.y_hat_not_zero), self.y_not_zero)).mean(), 3)
        self.mae = round(mean_absolute_error(self.y, self.y_hat), 3)
        self.rmae = round(self.mae/self.mean_y, 3)
        self.rmse = round(math.sqrt(mean_squared_error(self.y, self.y_hat)), 3)
        #self.std_rmse = np.std(self)
        self.rrmse = round(self.rmse/self.mean_y, 3)
        self.bias = round(np.mean(self.y - self.y_hat), 3)

        return self.re, self.mae, self.rmae, self.rmse, self.rrmse, self.bias

    def MeanLoss(self, delta):
        difference_abs = np.absolute(self.y_hat - self.y) - np.full(len(self.y), delta)
        difference_abs[np.where(difference_abs < 0)[0]] = 0
        self.meanLoss = round(np.mean(difference_abs), 3)
        self.RmeanLoss = round(self.meanLoss/self.mean_y, 3)
        return self.meanLoss, self.RmeanLoss

