# import necessary modules

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def splitter(df: pd.DataFrame, steps_in: int, steps_out: int):
    """
    :param df:  pandas DataFrame to split
    :param steps_in: number of values in the in data
    :param steps_out:  number of values in the out data
    :return: tuple (X,y) of examples
    """
    (index, series) = tuple(df.shape)
    X = list()
    y = list()
    sample_size = steps_in + steps_out
    for i in range(index-sample_size):
        # define the bounds
        out_left = i + steps_in
        out_right = out_left + steps_out
        in_right = i + steps_in
        in_left = i

        # extract the data
        in_values = df.iloc[in_left:in_right, :].values
        out_values = df.iloc[out_left:out_right, :].values

        # reshape
        X_values = np.array(in_values).reshape((len(in_values)*series,))
        y_values = np.array(out_values).reshape((len(out_values) * series,))

        X.append(X_values)
        y.append(y_values)

    return np.array(X), np.array(y)


def to_predictable(df: pd.DataFrame):
    """
    transform a dataframe to an array ready to be used by the regression models
    :param df:
    :return:
    """
    (index, series) = tuple(df.shape)
    values = df.values
    values = np.array(values).reshape((len(values)*series,))
    return values

def to_dataframe(ar, shape, index):
    """
    transfrom an array to a dataframe
    :param ar:
    :param shape:
    :return:
    """
    return pd.DataFrame(ar.reshape(shape), index=index)


def smape_loss(y_true, y_pred):
    """
    Error function
    :param y_true:
    :param y_pred:
    :return:
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


def fit_and_predict(df,
                    df_to_pred,
                    model,
                    steps_in,
                    steps_out,
                    pred_index,
                    loss_function,
                    test_size=0.3,
                    random_state=314):
    (X, y) = splitter(df, steps_in, steps_out)
    (X_train, X_test, y_train, y_test) = \
        train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)

    df_pred = to_dataframe()
