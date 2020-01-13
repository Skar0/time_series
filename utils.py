import glob
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


def split_sequence_nn_with_past_multi_step(sequence, n_steps_in, n_steps_out):
    """
    Splits the sequence into lists of input sample and data to be predicted.
    Example: if n_steps_in = 7 and  n_steps_out = 2 we create an input consisting of the 2 days we want to predict but
    from the past year and the 7 previous days and 2 days to be predicted.

    :param sequence:
    :param n_steps_in:
    :param n_steps_out:
    :return:
    """

    X, y = list(), list()
    for i in range(365 - n_steps_in, len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break

        # gather input and output parts of the pattern
        seq_x, seq_y = np.append(sequence[end_ix - 365:out_end_ix - 365], sequence[i:end_ix]), sequence[
                                                                                               end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_sequence_nn_with_past_outliers_multi_step(sequence, sequence_with_outliers, n_steps_in, n_steps_out):
    """
    From https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/
    Splits the sequence into lists of input sample and data to be predicted.
    Example: if n_steps_in = 7 and  n_steps_out = 2 we create an input consisting of the 2 days we want to predict but
    from the past year and from the sequence containing outliers and the 7 previous days and 2 days to be predicted.

    :param sequence:
    :param sequence_with_outliers:
    :param n_steps_in:
    :param n_steps_out:
    :return:
    """

    X, y = list(), list()
    for i in range(365 - n_steps_in, len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break

        # gather input and output parts of the pattern
        seq_x, seq_y = np.append(np.array(sequence_with_outliers[end_ix - 365:out_end_ix - 365]),
                                 np.array(sequence[i:end_ix])), sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def weighted_average_of_submissions(input_path, output_path, models_to_try, smape_scores, auto_arima_forecast):
    """
    Performs a weighted sum of the different models tried for each series. The sum is weighted by the smape score of
    each model on their validation series. This creates a new submission file in output_path.

    csv files formatted according to find_submission_models.compute_and_plot_submission_with_arima should be contained
    in input_path. models_to_try should be a list of all models tried by this function for each series. smape_scores
    should should be a list of all models smapes on their validation series for each series. auto_arima_forecast should
    be the arima function used (by default arima_forecasting.auto_arima_forecast).

    :param input_path:
    :param output_path:
    :param models_to_try:
    :param smape_scores:
    :param auto_arima_forecast:
    :return:
    """

    # this will contain the final submission
    submission_dataframe = pd.DataFrame()

    # for each series
    for i in range(1, 46):

        # create a temporary dataframe
        temp = pd.DataFrame()

        # the models for this series
        methods = models_to_try[i - 1]

        # add arima to these methods (arima is always tried first, so we prepend)
        methods.insert(0, auto_arima_forecast)

        # smape scores
        smapes = smape_scores[i - 1]

        total_smape = sum(smapes)

        nbr_methods = len(methods)

        # for each method
        for j in range(nbr_methods):
            current_method = methods[j]

            # get the file for this method, file name format is known so we use a regex
            for file in glob.glob(str(input_path) + str(i) + "_formatted_" + current_method.__name__ + "*"):

                # if this is the first method
                if j == 0:

                    loaded = pd.read_csv(file, index_col='Id')
                    print("method " + current_method.__name__ + " " + str(smapes[j]))
                    print(loaded.head())
                    print()
                    temp['forecasts'] = loaded['forecasts'] * (
                        (1 / (nbr_methods - 1)) * (1 - (smapes[j] / total_smape)))

                else:

                    loaded = pd.read_csv(file, index_col='Id')
                    print("method " + current_method.__name__ + " " + str(smapes[j]))
                    print(loaded.head())
                    print()
                    temp['forecasts'] = temp['forecasts'] + (
                        loaded['forecasts'] * ((1 / (nbr_methods - 1)) * (1 - (smapes[j] / total_smape))))

        print(temp['forecasts'].head())
        print("----------------------------------------------------------------------" + str(nbr_methods))
        submission_dataframe = submission_dataframe.append(temp)

    print("submission")
    submission_dataframe.to_csv(output_path)
    print(submission_dataframe.to_string())
