import os
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.getLogger().setLevel(os.environ.get("LOG_LEVEL", "INFO"))
logging.basicConfig(level=logging.INFO)


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
    def __init__(self, model: pd.Series = None):
        self.theta = None
        if model is not None:
            self.theta = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. For the BaselineModel, it will compute the parameters theta, a pd.Series with the average time spent in the vehicle for each transport type.
        :param X_train: pd.DataFrame with the features.
        :return self: the fitted model.
        '''
        logging.debug("Train dataset: X: ", X_train.shape, "y: ", y_train.shape)
        velocity = {}
        for name, group in X_train.groupby('transport'):
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
    def __init__(self, model: pd.Series = None):
        self.theta = None
        if model is not None:
            self.theta = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. For the BaselineModel, it will compute the parameters theta, a pd.Series with the average time spent in the vehicle for each transport type.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''
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


def _encode_timestamps_dummy_variables(X: pd.DataFrame):
    '''
    Function to encode the timestamps as dummy variables.
    :param X: pd.DataFrame with the features.
    :return:
    '''

    # Perform one-hot encoding
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    timestamp_cols = [col for col in X.columns if 'timestamp' in col]
    X_encoded = X.copy()
    X_encoded.loc[:, 'creation_date_year'] = X_encoded['creation_timestamp'].dt.year
    X_encoded.loc[:, 'creation_date_month'] = X_encoded['creation_timestamp'].dt.month
    X_encoded.loc[:, 'creation_date_day'] = X_encoded['creation_timestamp'].dt.day
    for col in timestamp_cols:
        X_encoded.loc[:, f'{col}_hour'] = X_encoded[col].dt.hour
        X_encoded.loc[:, f'{col}_minute'] = X_encoded[col].dt.minute
        X_encoded.loc[:, f'{col}_second'] = X_encoded[col].dt.second

    X_encoded.drop(columns=timestamp_cols, inplace=True)
    X_encoded = pd.get_dummies(X_encoded, drop_first=True)

    # Ensure all data is numerical
    X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
    return X_encoded


class LinearModel_encode_timestamps_dummy_variables(Estimator):
    '''
    Linear model that predicts the time from pickup to delivery.
    It computes the linear regression model using the features as input.
    '''
    def __init__(self, model: LinearRegression() = None):
        self.model = LinearRegression()
        if model is not None:
            self.theta = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. It will compute the linear regression model.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''

        # Perform one-hot encoding
        logging.info("Starting to encode variables")
        X_encoded = _encode_timestamps_dummy_variables(X_train)
        logging.info("Finished to encode variables. Starting to fit the model")

        # Fit the model
        self.model.fit(X_encoded, y_train)
        logging.info("Finished training the model")
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        # Perform one-hot encoding
        X_encoded = _encode_timestamps_dummy_variables(X)

        # Align columns with the training data
        X_encoded = X_encoded.reindex(columns=self.model.feature_names_in_, fill_value=0)

        return self.model.predict(X_encoded)

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


class LinearModelSGD_encode_timestamps_dummy_variables(Estimator):
    '''
    Linear model that predicts the time from pickup to delivery.
    It computes the linear regression model using the features as input.
    '''
    def __init__(self, model: LinearRegression() = None):
        self.model = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
        if model is not None:
            self.theta = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. It will compute the linear regression model.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''

        # Perform one-hot encoding
        logging.info("Starting to encode variables")
        X_encoded = _encode_timestamps_dummy_variables(X_train)
        logging.info("Finished to encode variables. Starting to fit the model")

        # Fit the model
        self.model.fit(X_encoded, y_train)
        logging.info("Finished training the model")
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        # Perform one-hot encoding
        X_encoded = _encode_timestamps_dummy_variables(X)

        # Align columns with the training data
        X_encoded = X_encoded.reindex(columns=self.model.feature_names_in_, fill_value=0)

        return self.model.predict(X_encoded)

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


class LinearModel_encode_timestamps_cyclical(Estimator):
    '''
    Linear model that predicts the time from pickup to delivery.
    It computes the linear regression model using the features as input.
    '''
    def __init__(self, model: LinearRegression() = None):
        self.model = LinearRegression()
        if model is not None:
            self.theta = model

    def _sin_transformer(self, period):
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def _cos_transformer(self, period):
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    def _encode_timestamps_cyclical(self, X: pd.DataFrame):
        '''
        Function to encode the timestamps as dummy variables.
        :param X: pd.DataFrame with the features.
        :return:
        '''

        # Perform one-hot encoding
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        timestamp_cols = [col for col in X.columns if 'timestamp' in col]
        X_encoded = X.copy()

        X_encoded["creation_date_day"] = X_encoded["creation_timestamp"].dt.day
        X_encoded.loc[:, 'creation_date_day_sin'] = self._sin_transformer(31).fit_transform(X_encoded[['creation_date_day']])
        X_encoded.loc[:, 'creation_date_day_cos'] = self._cos_transformer(31).fit_transform(X_encoded[['creation_date_day']])
        X_encoded.drop(columns="creation_date_day", inplace=True)

        X_encoded["creation_date_month"] = X_encoded["creation_timestamp"].dt.month
        X_encoded.loc[:, 'creation_date_month_sin'] = self._sin_transformer(12).fit_transform(X_encoded[['creation_date_month']])
        X_encoded.loc[:, 'creation_date_month_cos'] = self._cos_transformer(12).fit_transform(X_encoded[['creation_date_month']])
        X_encoded.drop(columns="creation_date_month", inplace=True)

        X_encoded["creation_date_year"] = X_encoded["creation_timestamp"].dt.year
        X_encoded.loc[:, 'creation_date_year_sin'] = self._sin_transformer(1).fit_transform(X_encoded[['creation_date_year']])
        X_encoded.loc[:, 'creation_date_year_cos'] = self._cos_transformer(1).fit_transform(X_encoded[['creation_date_year']])
        X_encoded.drop(columns="creation_date_year", inplace=True)

        for col in timestamp_cols:
            X_encoded.loc[:, f'{col}_hour'] = X_encoded[col].dt.hour
            X_encoded.loc[:, f'{col}_hour_sin'] = self._sin_transformer(24).fit_transform(X_encoded[[f'{col}_hour']])
            X_encoded.loc[:, f'{col}_hour_cos'] = self._cos_transformer(24).fit_transform(X_encoded[[f'{col}_hour']])
            X_encoded.drop(columns=f'{col}_hour', inplace=True)

            X_encoded.loc[:, f'{col}_minute'] = X_encoded[col].dt.minute
            X_encoded.loc[:, f'{col}_minute_sin'] = self._sin_transformer(60).fit_transform(X_encoded[[f'{col}_minute']])
            X_encoded.loc[:, f'{col}_minute_cos'] = self._cos_transformer(60).fit_transform(X_encoded[[f'{col}_minute']])
            X_encoded.drop(columns=f'{col}_minute', inplace=True)

            X_encoded.loc[:, f'{col}_second'] = X_encoded[col].dt.second
            X_encoded.loc[:, f'{col}_second_sin'] = self._sin_transformer(60).fit_transform(X_encoded[[f'{col}_second']])
            X_encoded.loc[:, f'{col}_second_cos'] = self._cos_transformer(60).fit_transform(X_encoded[[f'{col}_second']])
            X_encoded.drop(columns=f'{col}_second', inplace=True)

        X_encoded.drop(columns=timestamp_cols, inplace=True)
        X_encoded = pd.get_dummies(X_encoded, drop_first=True)

        # Ensure all data is numerical
        X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
        return X_encoded

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. It will compute the linear regression model.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''

        logging.info("Starting to encode variables")
        X_encoded = self._encode_timestamps_cyclical(X_train)
        logging.info("Finished to encode variables. Starting to fit the model")

        # Fit the model
        self.model.fit(X_encoded, y_train)
        logging.info("Finished training the model")
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        # Perform one-hot encoding
        X_encoded = self._encode_timestamps_cyclical(X)

        # Align columns with the training data
        X_encoded = X_encoded.reindex(columns=self.model.feature_names_in_, fill_value=0)

        return self.model.predict(X_encoded)

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

