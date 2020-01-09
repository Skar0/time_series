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


def to_predictable(df: pd.DataFrame, steps_in):
    """
    transform a dataframe to an array ready to be used by the regression models
    :param df:
    :return:
    """
    lenght = len(df.index)
    (index, series) = tuple(df.shape)
    k = lenght // steps_in
    l = []
    for i in range(k):
        values = np.array(df.loc[df.index[k*steps_in:(k+1)*steps_in]].values)
        values = values.reshape((len(values)*series,))
        l.append(values)

    return np.array(l)



def to_dataframe(ar, shape, index):
    """
    transfrom an array to a dataframe
    :param ar:
    :param shape:
    :return:
    """
    return pd.DataFrame(ar.reshape(shape), index=index)


def find_bounds_and_split(df: pd.DataFrame, steps_in, steps_out,
                          validation_size=0.2):
    length_df = len(df.index)

    # create size that are multiples of our steps_in and steps_out
    val_size = int(length_df*validation_size) - \
               (int(length_df*validation_size)) % steps_in

    train_df = df.head(length_df - val_size).copy()
    val_df = df.tail(val_size).copy()

    (X, y) = splitter(train_df, steps_in, steps_out)
    val_data = to_predictable(val_df, steps_in)

    # store everything in a dictionnary and return
    struct = {}
    struct['train_df'] = train_df
    struct['val_df'] = val_df
    struct['shape'] = tuple(df.index)
    struct['X'] = X
    struct['y'] = y
    struct['val_index'] = val_df.index
    struct['val_data'] = val_data

    return struct

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


def fit_and_predict(df: pd.DataFrame, df_to_pred, index_to_pred, model,
                    loss_function, steps_in, steps_out, validation_size=0.2):

    # recover everything with the struct
    struct = find_bounds_and_split(df, steps_in, steps_out,
                                   validation_size=validation_size)

    model.fit(struct['X'], struct['y'])

    val_data = struct['val_data']
    val_df = struct['val_df']
    val_df_ar = to_predictable(val_df, steps_in)
    val_pred = model.predict(val_data)
    val_pred_df = to_dataframe(val_pred, tuple(val_df.shape), val_df.index)
    score = loss_function(val_df_ar, val_pred)

    ar_to_pred = to_predictable(df_to_pred, steps_in)
    ar_pred = model.predict(ar_to_pred)

    shape = (steps_out, struct['shape'][1])

    df_pred = to_dataframe(ar_pred, shape, index_to_pred)

    return (df_pred, val_pred_df, score)
