import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

# Estimator will be an abstract class, and then we will define our different estimators like the baseline model or the linear model as subclasses of Estimator.
class Estimator(ABC):
    '''
    Abstract class for the different estimators that we will use in the project.
    '''

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data.
        :return self: the fitted model.
        '''
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :return: y_hat: pd.Series with the predicted values.
        '''
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        '''
        Method to evaluate the model.
        It will compute the loss, a numerical metric that describes how wrong a model's predictions are.
        Loss measures the distance between the model's predictions and the actual labels.
        :param X_test: pd.DataFrame with the features.
        :param y_test: pd.Series with the target variable.
        :return: loss: float with the loss of the model.
        '''
        pass


class BaselineModel_sum(Estimator):
    '''
    Baseline model that predicts the time from pickup to delivery.
    It computes the velocity for each transport type as the sum of the pickup to delivery distance (computed using the haversine distance) divided by the sum of the pickup to delivery (PD) time.
    This is not the best way to compute the velocity, as we are loosing variability info about velocity for each order.
    '''
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.theta = None

    def dataset_shape(self):
        '''
        Method to print the shape of the train and test datasets.
        :return: shape of the train and test datasets.
        '''
        print("Train dataset", self.X_train.shape, self.y_train.shape)
        print("Test dataset", self.X_test.shape, self.y_test.shape)
        return self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. For the BaselineModel, it will compute the parameters theta, a pd.Series with the average time spent in the vehicle for each transport type.
        :param X_train: pd.DataFrame with the features.
        :return self: the fitted model.
        '''
        self.X_train = X_train
        self.y_train = y_train
        velocity = {}
        for name, group in self.X_train.groupby('transport'):
            velocity[name] = sum(group['pd_distance_haversine_m']) / sum((group['delivery_entering_timestamp'] - group['pickup_timestamp']).dt.total_seconds())

        velocity = pd.Series(velocity, name='velocity (m/s)')
        self.theta = velocity
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        y_hat = []
        if isinstance(X, pd.Series):
            y_hat.append(X['pd_distance_haversine_m'] / self.theta[X['transport']])
        if isinstance(X, pd.DataFrame):
            for index, row in X.iterrows():
                y_hat.append(row['pd_distance_haversine_m'] / self.theta[row['transport']])
        y_hat = pd.Series(y_hat, dtype=np.float64, name='pickup_to_delivery_predicted')
        return y_hat

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        '''
        Method to evaluate the model. It will compute the mean absolute error (MAE) and the mean squared error (MSE).
        :param X_test: pd.DataFrame with the features.
        :param y_test: pd.Series with the target variable.
        :return: mae: float with the mean absolute error.
        :return: mse: float with the mean squared error.
        '''
        y_hat = self.predict(X_test)
        print("y_hat", type(y_hat), y_hat.dtype)
        print("y_test", type(y_test), y_test.dtype)
        mae = np.mean(np.abs(y_test - y_hat))
        mse = np.mean((y_test - y_hat)**2)
        return mae, mse



class BaselineModel_mean(Estimator):
    '''
    Baseline model that predicts the time from pickup to delivery.
    It computes the velocity for each transport type and for each order, and then computes the mean of the velocity for each transport type.
    '''
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.theta = None

    def dataset_shape(self):
        '''
        Method to print the shape of the train and test datasets.
        :return: shape of the train and test datasets.
        '''
        print("Train dataset", self.X_train.shape, self.y_train.shape)
        print("Test dataset", self.X_test.shape, self.y_test.shape)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. For the BaselineModel, it will compute the parameters theta, a pd.Series with the average time spent in the vehicle for each transport type.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''
        self.X_train = X_train
        self.y_train = y_train
        X_train['time'] = (X_train['delivery_entering_timestamp'] - X_train['pickup_timestamp']).dt.total_seconds()
        X_train['velocity'] = X_train['pd_distance_haversine_m'] / X_train['time']
        # For few rows we have that delivery_entering_timestamp = pickup_timestamp, so the velocity is infinite. We will drop the rows where the velocity is infinite
        X_new = X_train[X_train['velocity'] != np.inf].copy()
        # Differentiate the velocity per vehicle type
        velocity = X_new.groupby(X_new['transport'])['velocity'].mean()
        self.theta = velocity
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        y_hat = []
        if isinstance(X, pd.Series):
            y_hat.append(X['pd_distance_haversine_m'] / self.theta[X['transport']])
        if isinstance(X, pd.DataFrame):
            for index, row in X.iterrows():
                y_hat.append(row['pd_distance_haversine_m'] / self.theta[row['transport']])
        y_hat = pd.Series(y_hat, dtype=np.float64, name='pickup_to_delivery_predicted')
        return y_hat

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        '''
        Method to evaluate the model. It will compute the mean absolute error (MAE) and the mean squared error (MSE).
        :param X_test: pd.DataFrame with the features.
        :param y_test: pd.Series with the target variable.
        :return: mae: float with the mean absolute error.
        :return: mse: float with the mean squared error.
        '''
        y_hat = self.predict(X_test)
        mae = np.mean(np.abs(y_test - y_hat))
        mse = np.mean((y_test - y_hat)**2)
        return mae, mse


class LinearModel(Estimator):
    '''
    Linear model that predicts the time from pickup to delivery.
    It computes the linear regression model using the features as input.
    '''
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.theta = None

    def dataset_shape(self):
        '''
        Method to print the shape of the train and test datasets.
        :return: shape of the train and test datasets.
        '''
        print("Train dataset", self.X_train.shape, self.y_train.shape)
        print("Test dataset", self.X_test.shape, self.y_test.shape)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. It will compute the linear regression model.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''
        self.X_train = X_train
        self.y_train = y_train
        reg = LinearRegression().fit(X_train, y_train)
        self.theta = reg
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        return self.theta.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        '''
        Method to evaluate the model. It will compute the mean absolute error (MAE) and the mean squared error (MSE).
        :param X_test: pd.DataFrame with the features.
        :param y_test: pd.Series with the target variable.
        :return: mae: float with the mean absolute error.
        :return: mse: float with the mean squared error.
        '''
        y_hat = self.predict(X_test)
        mae = np.mean(np.abs(y_test - y_hat))
        mse = np.mean((y_test - y_hat)**2)
        return mae, mse

