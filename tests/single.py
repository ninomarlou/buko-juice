import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from context import aiwf, test_data_1, test_data_2, test_data_3, test_data_4
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def main():
    # Simple Linear Regression

    # Importing the libraries

    # Importing the dataset
    size_of_test = .7
    dataset = test_data_4
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Encoding categorical data
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_of_test, random_state=0)

    # Training the Random Forest Regression model on the whole dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)

    # Predicting a new result
    y_pred = regressor.predict(X_test)

    evs = explained_variance_score(y_test, y_pred)
    me = max_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    r2vw = r2_score(y_test, y_pred, multioutput='variance_weighted')
    r2ua = r2_score(y_test, y_pred, multioutput='uniform_average')
    r2rv = r2_score(y_test, y_pred, multioutput='raw_values')

    score = [[evs, me, mae, mse, msle, mape, mdae, r2vw, r2ua, r2rv]]
    column_names = ['EVS', 'ME', 'MAE', 'MSE', 'MSLE', 'MAPE', 'MDAE', 'R2 (VW)', 'R2 (UA)', 'R2 (RV)']
    df_scores = pd.DataFrame(score, columns=column_names)
    display_scores(df_scores)
    # display_scores(df_scores)


def display_scores(df):
    pd.set_option('display.max_columns', 11)
    pd.set_option('display.width', 2000)
    def pdtabulate(df): return tabulate(df, headers='keys', tablefmt='psql', showindex=False)
    print(pdtabulate(df))

    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')

    # Visualising the Random Forest Regression results (higher resolution)


if __name__ == '__main__':
    main()
