import os
import mlflow, mlflow.sklearn
import argparse
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


def load_data():
    df = pd.read_csv('wine_quality.csv')
    df.drop(columns='Id', inplace=True)
    print(df.head())
    print(df.columns)
    return df

def eval(actual, predicted):
    rmse = root_mean_squared_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    return rmse, mape, r2, mae


def main(alpha, l1_ratio):
    df = load_data()
    TARGET = 'quality'

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=25)
    
    mlflow.set_experiment('ML-Model-1')
    with mlflow.start_run():
        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse, mape, r2, mae = eval(y_test, y_pred)

        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mape', mape)
        mlflow.log_metric('r2-score', r2)
        mlflow.log_metric('mae', mae)

        mlflow.sklearn.log_model(model, 'ElasticNet-Trained') # model, folder name



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--alpha', '-a', type=float, default=0.2)
    args.add_argument('--l1_ratio', '-l1', type=float, default=0.3)
    parsed_args = args.parse_args()

    # passing the arguments to the function
    main(parsed_args.alpha, parsed_args.l1_ratio)