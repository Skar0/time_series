import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# transform a dataframe to the corresponding numpy array
def to_array(f_df: pd.DataFrame):
    f_ar = []
    for f_column in f_df.columns:
        v = f_df[f_column].values
        v = np.array(v)
        v = v.reshape((len(v),))
        f_ar.append(v)

    return np.array(f_ar)


# smape loss function to compute the score
def smape_loss(f_y_true, f_y_pred):
    """
    Error function
    :param f_y_true:
    :param f_y_pred:
    :return:
    """

    denominator = (np.abs(f_y_true) + np.abs(f_y_pred)) / 200.0
    diff = np.abs(f_y_true - f_y_pred) / denominator
    diff[denominator == 0] = 0.0

    return np.mean(diff)


# this function create a new index that follows the last day in the dataframe
# f_df and has length f_steps_out
def create_next_index(f_df: pd.DataFrame, f_steps_out):
    last_day = f_df.index[-1]
    prob_index_range = pd.date_range(start=last_day, periods=f_steps_out+1)
    res_index_range = prob_index_range[1:]

    return res_index_range


# tool function to rename the columns of a fresh dataframe
def create_rename_dic():
    rn_dic = {}

    for f_i in range(45):  # TODO: has a magic number, to be changed
        rn_dic[f_i] = "series-" + str(f_i+1)

    return rn_dic

################################################################################
################################################################################


# I'm using this instead of train_test_split of sklearn, I forgot the motive
def split_to_test(f_X, f_y, f_days, f_series, f_steps_in, f_steps_out, f_test_size):
    """
    split dataset leaving the order of examples untouched
    :param f_X:
    :param f_y:
    :param f_days:
    :param f_series:
    :param f_steps_in:
    :param f_steps_out:
    :param f_test_size:
    :return:
    """
    regression_length = f_days - (f_steps_in + f_steps_out)
    f_train_size = int(regression_length * (1 - f_test_size))
    # f_tail_size = regression_length - f_train_size

    res_dataset = []

    # do for all series
    for f_s in range(f_series):
        t = [f_X[f_s][:f_train_size], f_X[f_s][f_train_size:], f_y[f_s][:f_train_size],
             f_y[f_s][f_train_size:]]
        res_dataset.append(t)

    res_dataset = np.array(res_dataset)

    return res_dataset


def split_to_test2(f_X, f_y, f_days, f_series, f_steps_in, f_steps_out, f_test_size=0.2,
                   f_random_state=314):
    """
    split dataset leaving the order of examples untouched
    :param f_X:
    :param f_y:
    :param f_days:
    :param f_series:
    :param f_steps_in:
    :param f_steps_out:
    :param f_test_size:
    :param f_random_state:
    :return:
    """
    regression_length = f_days - (f_steps_in + f_steps_out)
    f_train_size = int(regression_length * (1 - f_test_size))
    # f_tail_size = regression_length - f_train_size

    res_dataset = []

    # do for all series
    for f_s in range(f_series):
        res_X_train, res_X_test, res_y_train, res_y_test = train_test_split(f_X[f_s], f_y[f_s],
                                                                            test_size=f_test_size,
                                                                            random_state=f_random_state)

        t = [res_X_train, res_X_test, res_y_train, res_y_test]
        res_dataset.append(t)

    res_dataset = np.array(res_dataset)

    return res_dataset

################################################################################
################################################################################


# compute the score of a prediction with respect to f_loss_function
def compute_score(f_real_df, f_pred, f_steps_out, f_loss_function):
    """
    creates an array with the smape values by series present in real_df
    :param f_real_df:
    :param f_pred:
    :param f_steps_out:
    :param f_loss_function:
    :return:
    """
    (f_days, f_series) = tuple(f_real_df.shape)

    res_scores = np.zeros((f_series,))

    f_real_array = to_array(f_real_df.tail(f_steps_out))
    f_pred_array = to_array(f_pred)

    for f_s in range(f_series):
        res_scores[f_s] = f_loss_function(f_real_array[f_s, :], f_pred_array[f_s, :])

    return res_scores


# creates a new prediction, explained how below
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
