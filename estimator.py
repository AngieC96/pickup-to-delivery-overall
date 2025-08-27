import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

logging.getLogger().setLevel(os.environ.get("LOG_LEVEL", "INFO"))
logging.basicConfig(level=logging.INFO) # USe INFO to see more informations, WARNING to see less


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

    def test(self, X_test: pd.DataFrame, y_test: pd.Series):
        '''
        Method to test the model on the test set.
        :param X_test: pd.DataFrame with the features.
        :param y_test: pd.Series with the target variable.
        :return: results: pd.DataFrame with the predictions and the actual values.
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
        self.model = None
        if model is not None:
            self.model = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. For the BaselineModel, it will compute the parameters model, a pd.Series with the average time spent in the vehicle for each transport type.
        :param X_train: pd.DataFrame with the features.
        :return self: the fitted model.
        '''
        logging.info(f"Train datasets shapes: X: {X_train.shape}, y: {y_train.shape}")
        velocity = {}
        for name, X_group in X_train.groupby('transport'):
            y_group = y_train[X_group.index]
            velocity[name] = sum(X_group['pd_distance_haversine_m']) / sum(y_group)

        velocity = pd.Series(velocity, name='velocity (m/s)')
        self.model = velocity
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        y_hat = []
        if isinstance(X, pd.Series):
            y_hat.append(X['pd_distance_haversine_m'] / self.model[X['transport']])
        if isinstance(X, pd.DataFrame):
            for index, row in X.iterrows():
                y_hat.append(row['pd_distance_haversine_m'] / self.model[row['transport']])
        y_hat = pd.Series(y_hat, dtype=np.float64, name='pickup_to_delivery_predicted')
        return y_hat
    
    def test(self, X_test: pd.DataFrame, y_test: pd.Series):
        results_dataset = pd.concat([
            X_test,
            y_test.rename("target"),
            pd.Series(self.predict(X_test), name="prediction"),
        ], axis=1)
        results_dataset['residual'] = results_dataset['prediction'] - results_dataset['target']
        return results_dataset

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
        r2 = r2_score(y_test, y_hat)
        return mae, mse, r2



class BaselineModel_mean(Estimator):
    '''
    Baseline model that predicts the time from pickup to delivery.
    It computes the velocity for each transport type and for each order, and then computes the mean of the velocity for each transport type.
    '''
    def __init__(self, model: pd.Series = None):
        self.model = None
        if model is not None:
            self.model = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. For the BaselineModel, it will compute the parameters model, a pd.Series with the average time spent in the vehicle for each transport type.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''
        logging.info(f"Train datasets shapes: X: {X_train.shape}, y: {y_train.shape}")
        X_train['velocity'] = X_train['pd_distance_haversine_m'] / y_train
        # For few rows we have that delivery_timestamp = pickup_timestamp, so the velocity is infinite. We will drop the rows where the velocity is infinite
        X_new = X_train[X_train['velocity'] != np.inf].copy()
        # Differentiate the velocity per vehicle type
        velocity = X_new.groupby(X_new['transport'])['velocity'].mean()
        self.model = velocity
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        y_hat = []
        if isinstance(X, pd.Series):
            y_hat.append(X['pd_distance_haversine_m'] / self.model[X['transport']])
        if isinstance(X, pd.DataFrame):
            for index, row in X.iterrows():
                y_hat.append(row['pd_distance_haversine_m'] / self.model[row['transport']])
        y_hat = pd.Series(y_hat, dtype=np.float64, name='pickup_to_delivery_predicted')
        return y_hat
    
    def test(self, X_test: pd.DataFrame, y_test: pd.Series):
        results_dataset = pd.concat([
            X_test,
            y_test.rename("target"),
            pd.Series(self.predict(X_test), name="prediction"),
        ], axis=1)
        results_dataset['residual'] = results_dataset['prediction'] - results_dataset['target']
        return results_dataset

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
        r2 = r2_score(y_test, y_hat)
        return mae, mse, r2


def _encode_timestamps_dummy_variables(X: pd.DataFrame):
    '''
    Function to encode the timestamps as dummy variables.
    :param X: pd.DataFrame with the features.
    :return: X_encoded: pd.DataFrame with the encoded features.
    '''

    # Perform one-hot encoding
    X_encoded = X.copy()
    X_encoded.loc[:, 'activation_date_year'] = X_encoded['activation_timestamp'].dt.year
    X_encoded.loc[:, 'activation_date_month'] = X_encoded['activation_timestamp'].dt.month
    X_encoded.loc[:, 'activation_date_day'] = X_encoded['activation_timestamp'].dt.day
    X_encoded.loc[:, 'activation_date_weekday'] = X_encoded['activation_timestamp'].dt.weekday

    timestamp_cols = [col for col in X.columns if 'timestamp' in col]
    for col in timestamp_cols:
        X_encoded.loc[:, f'{col}_hour'] = X_encoded[col].dt.hour
        X_encoded.loc[:, f'{col}_minute'] = X_encoded[col].dt.minute
        X_encoded.loc[:, f'{col}_second'] = X_encoded[col].dt.second

    X_encoded.drop(columns=['activation_timestamp'], inplace=True)
    X_encoded = pd.get_dummies(X_encoded, drop_first=True)

    # Ensure all data is numerical
    X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
    return X_encoded

def _sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def _cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
    
def _encode_timestamps_cyclical(X: pd.DataFrame):
    '''
    Function to encode the timestamps as cyclical variables.
    :param X: pd.DataFrame with the features.
    :return: X_encoded: pd.DataFrame with the encoded features.
    '''
    X_encoded = X.copy()
    X_encoded["activation_date_year"] = X_encoded["activation_timestamp"].dt.year
    X_encoded.loc[:, 'activation_date_year_sin'] = _sin_transformer(1).fit_transform(X_encoded[['activation_date_year']])
    X_encoded.loc[:, 'activation_date_year_cos'] = _cos_transformer(1).fit_transform(X_encoded[['activation_date_year']])
    X_encoded.drop(columns="activation_date_year", inplace=True)

    X_encoded["activation_date_month"] = X_encoded["activation_timestamp"].dt.month
    X_encoded.loc[:, 'activation_date_month_sin'] = _sin_transformer(12).fit_transform(X_encoded[['activation_date_month']])
    X_encoded.loc[:, 'activation_date_month_cos'] = _cos_transformer(12).fit_transform(X_encoded[['activation_date_month']])
    X_encoded.drop(columns="activation_date_month", inplace=True)

    X_encoded["activation_date_day"] = X_encoded["activation_timestamp"].dt.day
    X_encoded.loc[:, 'activation_date_day_sin'] = _sin_transformer(31).fit_transform(X_encoded[['activation_date_day']])
    X_encoded.loc[:, 'activation_date_day_cos'] = _cos_transformer(31).fit_transform(X_encoded[['activation_date_day']])
    X_encoded.drop(columns="activation_date_day", inplace=True)

    X_encoded.loc[:, 'activation_date_weekday'] = X_encoded['activation_timestamp'].dt.weekday
    X_encoded.loc[:, 'activation_date_weekday_sin'] = _sin_transformer(31).fit_transform(X_encoded[['activation_date_weekday']])
    X_encoded.loc[:, 'activation_date_weekday_cos'] = _cos_transformer(31).fit_transform(X_encoded[['activation_date_weekday']])
    X_encoded.drop(columns="activation_date_weekday", inplace=True)

    timestamp_cols = [col for col in X.columns if 'timestamp' in col]
    for col in timestamp_cols:
        X_encoded.loc[:, f'{col}_hour'] = X_encoded[col].dt.hour
        X_encoded.loc[:, f'{col}_hour_sin'] = _sin_transformer(24).fit_transform(X_encoded[[f'{col}_hour']])
        X_encoded.loc[:, f'{col}_hour_cos'] = _cos_transformer(24).fit_transform(X_encoded[[f'{col}_hour']])
        X_encoded.drop(columns=f'{col}_hour', inplace=True)

        X_encoded.loc[:, f'{col}_minute'] = X_encoded[col].dt.minute
        X_encoded.loc[:, f'{col}_minute_sin'] = _sin_transformer(60).fit_transform(X_encoded[[f'{col}_minute']])
        X_encoded.loc[:, f'{col}_minute_cos'] = _cos_transformer(60).fit_transform(X_encoded[[f'{col}_minute']])
        X_encoded.drop(columns=f'{col}_minute', inplace=True)

        X_encoded.loc[:, f'{col}_second'] = X_encoded[col].dt.second
        X_encoded.loc[:, f'{col}_second_sin'] = _sin_transformer(60).fit_transform(X_encoded[[f'{col}_second']])
        X_encoded.loc[:, f'{col}_second_cos'] = _cos_transformer(60).fit_transform(X_encoded[[f'{col}_second']])
        X_encoded.drop(columns=f'{col}_second', inplace=True)

    X_encoded.drop(columns=timestamp_cols, inplace=True)
    X_encoded = pd.get_dummies(X_encoded, drop_first=True)

    # Ensure all data is numerical
    X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
    return X_encoded


class LinearModel(Estimator):
    '''
    Linear model that predicts the time from pickup to delivery.
    It computes a linear regression model using the features as input.

    It uses the sklearn library to compute the linear regression model.
    With the option model_type, you can choose between:
    - linear: it will use the LinearRegression from sklearn.
    - SGD or sgd: it will use the SGDRegressor from sklearn.

    It can perform different encodings:
    - dummy: it will encode the timestamps as dummy variables.
    - dummy+difference: it will encode the timestamps as dummy variables and compute the difference between the timestamps.
    - cyclical: it will encode the timestamps as cyclical variables.
    - cyclical+difference: it will encode the timestamps as cyclical variables and compute the difference between the timestamps.
    '''
    def __init__(self, model: LinearRegression = None, model_type = 'linear', encoding = 'dummy', standardize = False):
        if standardize == 'minmax':
                self.scaler = MinMaxScaler()
        elif standardize == 'stdscaler':
                self.scaler = StandardScaler()
        elif standardize is False:
            self.scaler = None
        else:
            raise ValueError(f"Unknown standardize type: {standardize}. Available standardize are: minmax, stdscaler")

        if model_type == 'linear':
            if standardize is False:
                self.model = LinearRegression()
            else:
                self.model = make_pipeline(self.scaler, 
                                           LinearRegression())
        elif model_type == 'SGD' or model_type == 'sgd':
            if standardize is False:
                self.model = SGDRegressor(max_iter=1000, tol=1e-3)
            else:
                self.model = make_pipeline(self.scaler,
                                        SGDRegressor(max_iter=1000, tol=1e-3))
        else:
            raise ValueError(f"Unknown model type: {model_type}. Available models are: linear, SGD")
        
        if model is not None:
            if self.model_type == 'linear':
                assert isinstance(self.model, LinearRegression), f"The model provided must be a LinearRegression instance in accordance to {model_type}"
            if self.model_type == 'SGD' or model_type == 'sgd':
                assert isinstance(self.model, SGDRegressor), f"The model provided must be a SGDRegressor instance in accordance to {model_type}"
            self.model = model
        
        if encoding in ['dummy', 'cyclical']:
            self.encoding = encoding
        else:
            raise ValueError(f"Unknown encoding type: {encoding}. Available encodings are: dummy, cyclical")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. It will compute the linear regression model.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''
        logging.info(f"Train datasets shapes: X: {X_train.shape}, y: {y_train.shape}")
        logging.info(f"Train datasets columns: {X_train.columns}")
        logging.info("Starting to encode variables")
        # Perform the selected encoding
        if self.encoding == 'dummy':
            X_encoded = _encode_timestamps_dummy_variables(X_train)
        elif self.encoding == 'cyclical':
            X_encoded = _encode_timestamps_cyclical(X_train)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding}")
        
        logging.info(f"Encoded dataset shape: X: {X_encoded.shape}")
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
        # Perform the selected encoding
        if self.encoding == 'dummy':
            X_encoded = _encode_timestamps_dummy_variables(X)
        elif self.encoding == 'cyclical':
            X_encoded = _encode_timestamps_cyclical(X)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding}")

        # Align columns with the training data
        X_encoded = X_encoded.reindex(columns=self.model.feature_names_in_, fill_value=0)

        return self.model.predict(X_encoded)
    
    def test(self, X_test: pd.DataFrame, y_test: pd.Series):
        results_dataset = pd.concat([
            X_test,
            y_test.rename("target"),
            pd.Series(self.predict(X_test), name="prediction"),
        ], axis=1)
        results_dataset['residual'] = results_dataset['prediction'] - results_dataset['target']
        return results_dataset

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
        r2 = r2_score(y_test, y_hat)
        return mae, mse, r2


class RegressionTreeMethod(Estimator):
    '''
    Decision Tree Regression model that predicts the time from pickup to delivery.
    It computes a decision tree regression model using the features as input.
    '''
    def __init__(self, model = None, model_type = 'tree', encoding = 'dummy', n_estimators = 100, max_iter = 100, **kwargs):
        self.model_type = model_type
        if model_type == 'tree':
            self.model = DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1)
        elif model_type == 'randomforest':
            self.model = RandomForestRegressor(n_estimators=n_estimators, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=5)
        elif model_type == 'gradientboosting':
            self.model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=n_estimators, min_samples_split=2, min_samples_leaf=1)
        elif model_type == 'histgradientboosting':
            self.model = HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=max_iter, max_leaf_nodes=15, max_depth=None, min_samples_leaf=20, early_stopping=False)
        elif model_type == 'xgboost':
            self.model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators)
        elif model_type == 'lightgbm':
            self.model = LGBMRegressor(objective='regression', n_estimators=n_estimators)
        elif model_type == 'catboost':
            self.model = CatBoostRegressor(loss_function='RMSE', iterations=max_iter, verbose=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Available models are: tree, randomforest, gradientboosting, histgradientboosting, xgboost, lightgbm and catboost")

        if model is not None:
            if model_type == 'tree':
                assert isinstance(model, DecisionTreeRegressor), f"The model provided must be a DecisionTreeRegressor instance in accordance to {model_type}"
            elif model_type == 'randomforest':
                assert isinstance(model, RandomForestRegressor), f"The model provided must be a RandomForestRegressor instance in accordance to {model_type}"
            elif model_type == 'gradientboosting':
                assert isinstance(model, GradientBoostingRegressor), f"The model provided must be a GradientBoostingRegressor instance in accordance to {model_type}"
            elif model_type == 'histgradientboosting':
                assert isinstance(model, HistGradientBoostingRegressor), f"The model provided must be a HistGradientBoostingRegressor instance in accordance to {model_type}"
            elif model_type == 'xgboost':
                assert isinstance(model, XGBRegressor), f"The model provided must be a XGBRegressor instance in accordance to {model_type}"
            elif model_type == 'lightgbm':
                assert isinstance(model, LGBMRegressor), f"The model provided must be a LGBMRegressor instance in accordance to {model_type}"
            elif model_type == 'catboost':
                assert isinstance(model, CatBoostRegressor), f"The model provided must be a CatBoostRegressor instance in accordance to {model_type}"
            self.model = model
        
        if encoding in ['dummy', 'cyclical']:
            self.encoding = encoding
        else:
            raise ValueError(f"Unknown encoding type: {encoding}. Available encodings are: dummy, cyclical")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Method to fit the model to the data. It will compute the decision tree regression model.
        :param X_train: pd.DataFrame with the features.
        :param y_train: pd.Series with the target variable.
        :return self: the fitted model.
        '''
        logging.info(f"Train datasets shapes: X: {X_train.shape}, y: {y_train.shape}")
        logging.info(f"Train datasets columns: {X_train.columns}")
        logging.info("Starting to encode variables")
        # Perform the selected encoding
        if self.encoding == 'dummy':
            X_encoded = _encode_timestamps_dummy_variables(X_train)
        elif self.encoding == 'cyclical':
            X_encoded = _encode_timestamps_cyclical(X_train)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding}")
        
        logging.info(f"Encoded dataset shape: X: {X_encoded.shape}")
        logging.info("Finished to encode variables. Starting to fit the model")
        self.model.fit(X_encoded, y_train)
        logging.info("Finished training the model")
        return self

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the target variable. The function will give you the actual predictions for all samples inputted.
        :param X: pd.DataFrame with the features.
        :return: y_hat: pd.Series with the predicted values.
        '''
        # Perform the selected encoding
        if self.encoding == 'dummy':
            X_encoded = _encode_timestamps_dummy_variables(X)
        elif self.encoding == 'cyclical':
            X_encoded = _encode_timestamps_cyclical(X)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding}")

        # Align columns with the training data
        if  self.model_type == 'catboost':
            X_encoded = X_encoded.reindex(columns=self.model.feature_names_, fill_value=0)
        else:
            X_encoded = X_encoded.reindex(columns=self.model.feature_names_in_, fill_value=0)

        return self.model.predict(X_encoded)
    
    def test(self, X_test: pd.DataFrame, y_test: pd.Series):
        results_dataset = pd.concat([
            X_test,
            y_test.rename("target"),
            pd.Series(self.predict(X_test), name="prediction"),
        ], axis=1)
        results_dataset['residual'] = results_dataset['prediction'] - results_dataset['target']
        return results_dataset

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
        r2 = r2_score(y_test, y_hat)
        return mae, mse, r2

    def plot_feature_importance(self, X: pd.DataFrame, n_features=9999):
        if self.encoding == 'dummy':
            X_encoded = _encode_timestamps_dummy_variables(X)
        elif self.encoding == 'cyclical':
            X_encoded = _encode_timestamps_cyclical(X)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding}")
        feature_importance = self.model.feature_importances_
        print(X_encoded.columns)
        print(feature_importance)
        importance_df = pd.DataFrame(
            {"Feature": X_encoded.columns, "Importance": feature_importance}
        ).sort_values(by="Importance", ascending=False)
        importance_df = importance_df.sort_values(by="Importance", ascending=False)[:n_features]

        plt.figure(figsize=(10, 5))
        plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Model Feature Importance")
        plt.show()