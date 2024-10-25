import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Estimator will be an abstract class, and then we will define our different estimators like the baseline model or the linear model as subclasses of Estimator.
class Estimator(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class BaselineModel_sum(Estimator):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def dataset_shape(self):
        print("Train dataset", self.X_train.shape, self.y_train.shape)
        print("Test dataset", self.X_test.shape, self.y_test.shape)

    def fit(self):
        velocity = {}
        for name, group in self.X_train.groupby('transport'):
            velocity[name] = sum(group['pd_distance_haversine_m']) / sum((group['delivery_entering_timestamp'] - group['pickup_timestamp']).dt.total_seconds())

        velocity = pd.Series(velocity, name='velocity (m/s)')
        return velocity

    def predict(self):
        print("I'll predict the delivery time using the velocity calculated in the fit method")



class BaselineModel_mean(Estimator):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def dataset_shape(self):
        print("Train dataset", self.X_train.shape, self.y_train.shape)
        print("Test dataset", self.X_test.shape, self.y_test.shape)

    def fit(self):
        X_train = self.X_train.copy()
        X_train['time'] = (X_train['delivery_entering_timestamp'] - X_train['pickup_timestamp']).dt.total_seconds()
        X_train['velocity'] = X_train['pd_distance_haversine_m'] / X_train['time']
        # For few rows we have that delivery_entering_timestamp = pickup_timestamp, so the velocity is infinite. We will drop the rows where the velocity is infinite
        X_new = X_train[X_train['velocity'] != np.inf].copy()
        # Differentiate the velocity per vehicle type
        velocity = X_new.groupby(X_new['transport'])['velocity'].mean()
        return velocity

    def predict(self):
        print("I'll predict the delivery time using the velocity calculated in the fit method")