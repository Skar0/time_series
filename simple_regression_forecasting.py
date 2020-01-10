# import necessary modules

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
################################################################################
# some preprocessing functions, must of all to remove outliers

def remove_q_outliers(data: pd.DataFrame, alpha=1.5):
    df = data.copy(deep=True)
    (ind, series) = tuple(df.shape)
    for s in range(series):
        serie = df.iloc[:, s]
        med = np.median(serie)  # most seen value
        q25, q75 = np.percentile(serie, 25), np.percentile(serie, 75)
        iqr = q75 - q25
        qoutliers1 = serie < q25 - alpha * iqr
        qoutliers2 = serie > q75 + alpha * iqr
        serie[qoutliers1] = np.nan
        serie[qoutliers2] = np.nan
        serie.fillna(med, inplace=True)
        df.iloc[:, s] = serie
    return df


def remove_z_outliers(data: pd.DataFrame, beta=3, gamma=0.6745):
    df = data.copy(deep=True)
    (ind, series) = tuple(df.shape)
    for s in range(series):
        serie = df.iloc[:, s]
        med = np.median(serie)  # most seen value
        mad = np.median(np.abs(serie - med))
        zoutlier = (gamma * (serie - med)) / mad > beta
        serie[zoutlier] = np.nan
        serie.fillna(med, inplace=True)
        df.iloc[:, s] = serie
    return df


# :/
def remove_outliers(data: pd.DataFrame, alpha=1.5, beta=3, gamma=0.6745):
    """
    this function removes outliers with two types of test:
    iqr test and z-score test, the z-score test has been modified to use the
    median, as the usual z-score is radically affected by outliers
    - alpha is a parameter for the iqr test
    - beta, gamma are for the z-mod-score test
    """
    df = data.copy(deep=True)
    (ind, series) = tuple(df.shape)
    for s in range(series):
        # extract the column
        serie = df.iloc[:, s]
        # compute needed stats
        q25, q75 = np.percentile(serie, 25), np.percentile(serie, 75)
        iqr = q75 - q25
        med = np.median(serie)  # most seen value
        mad = np.median(np.abs(serie - med))

        # z-score test
        zoutlier = (gamma * (serie - med)) / mad > beta

        # iqr outliers test
        qoutliers1 = serie < q25 - alpha * iqr
        qoutliers2 = serie > q75 + alpha * iqr

        # put and remove NaN values
        serie[zoutlier] = np.nan
        serie[qoutliers1] = np.nan
        serie[qoutliers2] = np.nan
        serie.fillna(med, inplace=True)
        df.iloc[:, s] = serie
    return df

################################################################################
################################################################################

def split_between_train_and_val(df: pd.DataFrame, steps_in, steps_out):
    train_df = df.head(len(df)-steps_in-steps_out).copy()
    val_df = df.tail(steps_in+steps_out).copy()
    val_df_X = val_df.head(steps_in).copy()
    val_df_y = val_df.tail(steps_out).copy()
    return (train_df, val_df_X, val_df_y)

def transfrom_to_list_of_df(df: pd.DataFrame):
    (index, series) = tuple(df.shape)
    l = []
    for ser in range(series):
        dfc = pd.DataFrame(df.iloc[:, ser]).copy()
        l.append(dfc)

    return l

def transform_to_list_of_tuples(df: pd.DataFrame, steps_in, steps_out):
    (index, series) = tuple(df.shape)

    (train_df, val_df_X, val_df_y) = split_between_train_and_val(df, steps_in, steps_out)

    l_train = transfrom_to_list_of_df(train_df)
    l_val_X = transfrom_to_list_of_df(val_df_X)
    l_val_y = transfrom_to_list_of_df(val_df_y)

    tuple_of_series = []

    for i in range(series):
        (X,y) = splitter(l_train[i], steps_in, steps_out)

        values_val_X =  l_val_X[i].values
        values_val_y = l_val_y[i].values

        validation_X =  values_val_X.reshape((len(values_val_X),))
        validation_y = values_val_y.reshape((len(values_val_y),))

        tuple_of_series.append([(X, y), (validation_X, validation_y)])

    return tuple_of_series


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


def train_for_model(lot, model):
    models = {}
    for i in range(len(lot)):
        m = model()
        [(X, y), (val_X, val_y)] = lot[i]
        m.fit(X, y)
        models[i] = m

    return models


def predict_horizon(df, models, steps_in, steps_out):
    (index, series) = tuple(df.shape)
    tail = df.tail(steps_in).copy()
    preds = []

    # predict for each series
    for ser in range(series):
        val = tail.iloc[:, ser].values
        val_X = np.array(val).reshape((len(val),))
        pred_y = models[ser].predict(val_X.reshape(1, -1))
        preds.append(pred_y)

    # create new index
    index_range = create_next_index(df, steps_out)

    # transfrom the predictions in a dataframe
    preds = np.array(preds)
    preds = preds.reshape((steps_out, series))
    preds = pd.DataFrame(preds, index=index_range)
    preds = preds.rename(columns=create_rename_dic())

    return preds


def main_by_model(df, model, steps_in, steps_out, loss_function):
    # get everything
    lot = transform_to_list_of_tuples(df, steps_in, steps_out)
    models = train_for_model(lot, model)

    score = []
    for i in range(len(lot)):
        [(X, y), (val_X, val_y)] = lot[i]
        val_X_pred = models[i].predict(val_X.reshape(1, -1))
        v = loss_function(val_X_pred, val_y)
        # print(v)
        score.append(v)

    preds = predict_horizon(df, models, steps_in, steps_out)
    preds = preds.rename(columns=create_rename_dic())

    return (preds, models, score)

def create_next_index(df: pd.DataFrame, steps_out):
    last_day = df.index[-1]
    prob_index_range = pd.date_range(start=last_day, periods=steps_out+1)
    index_range = prob_index_range[1:]

    return index_range

def predict_next(df: pd.DataFrame, models, steps_in, steps_out):
    (index, series) = tuple(df.shape)
    index_range = create_next_index(df, steps_out)

    if index == steps_in:
        preds = []

        for serie in range(series):
            # transform in predictable array
            val = df.iloc[:, serie].values
            val = np.array(val).reshape((len(val),))

            pred = models[serie].predict(val.reshape(1, -1))
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape((steps_out, series))

        preds = pd.DataFrame(preds, index=index_range)
        preds = preds.rename(columns=create_rename_dic())

        return preds

    return None

# this function is for this series
def create_rename_dic():
    rn_dic = {}

    for i in range(45):
        rn_dic[i] = "series-" + str(i+1)

    return rn_dic

def plot_by_series(df, models, steps_in, steps_out, figsize=(16,10)):

    val_df = df.loc[df.index[-(steps_in+steps_out):-steps_out]]
    val_df = predict_next(val_df, models, steps_in, steps_out)
    preds = predict_horizon(df, models, steps_in, steps_out)

    for column in df.columns:
        df_c = pd.DataFrame(df.tail(100)[column])
        val_df_c =  pd.DataFrame(val_df[column])
        preds_c = pd.DataFrame(preds[column])

        plt.figure(figsize=figsize)
        plt.plot(df_c, color="blue", linestyle="-")
        plt.plot(val_df_c, color="green", linestyle="-")
        plt.plot(preds_c, color="red", linestyle="--")
        plt.legend(["Train series", "Validation series", "Predicted series"])
        plt.title("Predictions using the LinearRegression model on " + str(column))

        plt.show()

def plot_all(df, models, steps_in, steps_out, figsize=(16,10)):

    val_df = df.loc[df.index[-(steps_in+steps_out):-steps_out]]
    val_df = predict_next(val_df, models, steps_in, steps_out)
    preds = predict_horizon(df, models, steps_in, steps_out)

    df_plot = df.tail(100)

    plt.figure(figsize=figsize)
    plt.plot(df_plot, color="blue", linestyle="-")
    plt.plot(val_df, color="green", linestyle="-")
    plt.plot(preds, color="red", linestyle="--")
    plt.legend(["Train series", "Validation series", "Predicted series"])
    plt.title("Predictions using the LinearRegression model")

    plt.show()
