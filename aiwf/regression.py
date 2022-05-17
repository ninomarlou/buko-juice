from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, r2_score
from .context import localisation
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class Regression:
    def __init__(self, size_of_test, X, y):
        self.sc = StandardScaler()
        self.size_of_test = size_of_test
        self.y = y
        self.X = self.sc.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=size_of_test, random_state=0)

    def run(self):
        scores = []
        simple_linear = Linear(self.X_train, self.X_test, self.y_train, self.y_test)
        scores.append(simple_linear.get_model_scores())

        support_vector = SupportVector(self.X_train, self.X_test, self.y_train, self.y_test)
        scores.append(support_vector.get_model_scores())

        decision_tree = DecisionTree(self.X_train, self.X_test, self.y_train, self.y_test)
        scores.append(decision_tree.get_model_scores())

        random_forest = RandomForest(self.X_train, self.X_test, self.y_train, self.y_test)
        scores.append(random_forest.get_model_scores())

        df_scores = pd.DataFrame(scores, columns=['Algorithm', 'EVS', 'ME', 'MAE', 'MSE', 'MSLE',
                                 'MAPE', 'MDAE', 'R2 (VW)', 'R2 (UA)', 'R2 (RV)'])
        self.display_scores(df_scores)

    def display_scores(self, df):
        pd.set_option('display.max_columns', 11)
        pd.set_option('display.width', 2000)
        print(localisation.messages.regression_result_message)
        def pdtabulate(df): return tabulate(df, headers='keys', tablefmt='psql', showindex=False)
        print(pdtabulate(df))
        print(localisation.messages.regression_legend)
        print('EVS     ', 'Explained variance score')
        print('ME      ', 'Max error')
        print('MAE     ', 'Mean absolute error')
        print('MSE     ', 'Mean squared error')
        print('MSLE    ', 'Mean squared log error')
        print('MAPE    ', 'Mean absolute percentage error')
        print('MDAE    ', 'Median absolute error')
        print('R2 (VW) ', 'R2 score (variance weighted)')
        print('R2 (UA) ', 'R2 score (uniform average)')
        print('R2 (RV) ', 'R2 score (raw values)')

        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')


class Linear:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Fit and test
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        self.y_pred = regressor.predict(self.X_test)

        # Metrics
        self.evs = explained_variance_score(self.y_test, self.y_pred)
        self.me = max_error(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.msle = mean_squared_log_error(self.y_test, self.y_pred)
        self.mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        self.mdae = median_absolute_error(self.y_test, self.y_pred)
        self.r2vw = r2_score(self.y_test, self.y_pred, multioutput='variance_weighted')
        self.r2ua = r2_score(self.y_test, self.y_pred, multioutput='uniform_average')
        self.r2rv = r2_score(self.y_test, self.y_pred, multioutput='raw_values')

    def get_model_scores(self):
        return ['Simple Linear', self.evs, self.me, self.mae, self.mse, self.msle, self.mape, self.mdae, self.r2vw, self.r2ua, self.r2rv]

    def regressor(self):
        return self.regressor


class SupportVector:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.reshape(-1, 1)
        self.y_test = y_test

        # Scale y
        self.sc = StandardScaler()
        self.y_train = self.sc.fit_transform(self.y_train)

        # Fit and test
        regressor = SVR(kernel='rbf')
        regressor.fit(self.X_train, self.y_train)
        self.y_pred_temp = regressor.predict(self.X_test).reshape(-1, 1)
        self.y_pred = self.sc.inverse_transform(self.y_pred_temp)

        # Metrics
        self.evs = explained_variance_score(self.y_test, self.y_pred)
        self.me = max_error(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.msle = mean_squared_log_error(self.y_test, self.y_pred)
        self.mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        self.mdae = median_absolute_error(self.y_test, self.y_pred)
        self.r2vw = r2_score(self.y_test, self.y_pred, multioutput='variance_weighted')
        self.r2ua = r2_score(self.y_test, self.y_pred, multioutput='uniform_average')
        self.r2rv = r2_score(self.y_test, self.y_pred, multioutput='raw_values')

    def get_model_scores(self):
        return ['Support vector', self.evs, self.me, self.mae, self.mse, self.msle, self.mape, self.mdae, self.r2vw, self.r2ua, self.r2rv]

    def regressor(self):
        return self.regressor


class DecisionTree:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Fit and test
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(self.X_train, self.y_train)
        self.y_pred = regressor.predict(self.X_test)

        # Metrics
        self.evs = explained_variance_score(self.y_test, self.y_pred)
        self.me = max_error(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.msle = mean_squared_log_error(self.y_test, self.y_pred)
        self.mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        self.mdae = median_absolute_error(self.y_test, self.y_pred)
        self.r2vw = r2_score(self.y_test, self.y_pred, multioutput='variance_weighted')
        self.r2ua = r2_score(self.y_test, self.y_pred, multioutput='uniform_average')
        self.r2rv = r2_score(self.y_test, self.y_pred, multioutput='raw_values')

    def get_model_scores(self):
        return ['Decision tree', self.evs, self.me, self.mae, self.mse, self.msle, self.mape, self.mdae, self.r2vw, self.r2ua, self.r2rv]

    def regressor(self):
        return self.regressor


class RandomForest:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Fit and test
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)
        regressor.fit(self.X_train, self.y_train)
        self.y_pred = regressor.predict(self.X_test)

        # Metrics
        self.evs = explained_variance_score(self.y_test, self.y_pred)
        self.me = max_error(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.msle = mean_squared_log_error(self.y_test, self.y_pred)
        self.mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        self.mdae = median_absolute_error(self.y_test, self.y_pred)
        self.r2vw = r2_score(self.y_test, self.y_pred, multioutput='variance_weighted')
        self.r2ua = r2_score(self.y_test, self.y_pred, multioutput='uniform_average')
        self.r2rv = r2_score(self.y_test, self.y_pred, multioutput='raw_values')

    def get_model_scores(self):
        return ['Randon forest', self.evs, self.me, self.mae, self.mse, self.msle, self.mape, self.mdae, self.r2vw, self.r2ua, self.r2rv]

    def regressor(self):
        return self.regressor
