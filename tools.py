import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def remove_outliers(series):
    """
    Removes outliers from the series and replaces their value by the median.
    This returns a new series.
    :param series:
    :return:
    """
    copy = series.copy()
    med = np.median(copy)
    q25, q75 = np.percentile(copy, 25), np.percentile(copy, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    upper = q75 + cut_off
    outliers = copy > upper
    copy[outliers] = np.nan
    copy.fillna(med, inplace=True)
    return copy


def normalize_series(series):
    """
    Normalize the series using the quantile method.
    This returns a new series and the scaler to perform inverse transform.
    :param series:
    :return:
    """
    # scale data
    scaler = MinMaxScaler()

    working_series = series.copy()
    # prepare data for normalization, i.e. put it in a matrix
    values = working_series.values
    values = values.reshape((len(values), 1))

    # fit and apply transformation
    normalized = scaler.fit_transform(values)

    # transform the scaled values as an array
    normalized = [val[0] for val in normalized]

    # create a series from the scaled values with correct index
    normalized_series = pd.Series(normalized, index=series.index)

    return scaler, normalized_series


def smape(y_true, y_pred):
    """
    Error function
    :param y_true:
    :param y_pred:
    :return:
    """
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


def keyvalue(df):
    """
    Transforms dataframe from the predictions in dataframe format (horizon x 45 i.e. one column of predictions per
    series) to the required format by Kaggle.
    :param df:
    :return:
    """
    df["horizon"] = range(1, df.shape[0] + 1)
    res = pd.melt(df, id_vars=["horizon"])
    res = res.rename(columns={"variable": "series"})
    res["Id"] = res.apply(lambda row: "s" + str(row["series"].split("-")[1]) + "h" + str(row["horizon"]), axis=1)
    res = res.drop(['series', 'horizon'], axis=1)
    res = res[["Id", "value"]]
    res = res.rename(columns={"value": "forecasts"})
    return res


def split_sequence(sequence, n_steps_in, n_steps_out):
    """
    From https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/
    Splits the sequence into lists of input sample and data to be predicted.
    Example: if n_steps_in = 7 and  n_steps_out = 2 we create 7 days as input and 2 days to be predicted.
    :param sequence:
    :param n_steps_in:
    :param n_steps_out:
    :return:
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def weighted_prediction(scores, preds):
    """
    suppose we used n methods for prediction
    format of entries:
        -scores: array of shape (n,45) where  score[i] is an array with 
            the score of method i for the 45 series
            -- by score of a method I mean the smape error
            and scores[i][j] is the score of method i for serie j+1 (mind
            the index)
        -preds : list of length n where preds[i] is a DATAFRAME that
            method i predicted, not a dataframe with only 1 series,
            the whole dataframe with 45 series inside, that is preds[i], 
            is ready for submission
    :return: 
    """

    if len(scores) == 1:  # only one method, return the series
        return preds[0]
    else:
        n = len(scores)
        # first compute total error by series
        total_error = {}
        for i in range(45):
            score = 0
            for j in range(n):
                score += scores[j][i]

            total_error[i] = score

        # now total_error is a dict with total_error[i] the total error
        # of all methods for the serie-(i+1)

        # create array of weight by method and series, being proportional
        # to the error
        normalizer = n-1
        weights = np.zeros(scores.shape)
        for i in range(n):
            for j in range(45):
                weights[i][j] = (1 / float(normalizer)) * (1 - scores[i][j]/total_error[j])

        # now weights[i][j] is the weights for serie j+1 given by method i

        # create a series to accumulate the mean of all methods
        df = pd.DataFrame(np.zeros(preds[0].shape), index=preds[0].index)

        # note that this series does not have the good name for the columns
        # we change that now
        rn = {}
        for i in range(45):
            rn[i] = 'series-' + str(i+1)
        df = df.rename(columns=rn)
        # now df has the good shape, the good name for the columns and zeros
        # everywhere

        for i in range(n):  # for each method
            for j in range(45):  # for each column
                column = 'series-' + str(j+1)

                # df[column] receives the weighted sum of values
                df[column] = df[column] + preds[i][column] * weights[i][j]

        return df

