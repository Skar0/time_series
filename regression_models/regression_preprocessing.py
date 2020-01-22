import pandas as pd
import numpy as np
from scipy import stats


def remove_outliers(f_df: pd.DataFrame, alpha=1.8, beta=3, gamma=0.6745, method='median',
                    bounds=False):
    """
    this function removes outliers with two types of test:
    iqr test and z-score test, the z-score test has been modified to use the
    median, as the usual z-score is radically affected by outliers
    - alpha is a parameter for the iqr test
    - beta, gamma are for the z-mod-score test
    - it implements as an addition to substitute the values to be replaced by the mode instead of
      the median
    """

    # copy the dataframe to leave to the user what to do with the original
    f_df_copy = f_df.copy(deep=True)
    (f_ind, f_series) = tuple(f_df_copy.shape)
    for f_s in range(f_series):

        # extract the column
        serie = f_df_copy.iloc[:, f_s]

        # compute needed stats
        q25, q75 = np.percentile(serie, 25), np.percentile(serie, 75)
        iqr = q75 - q25
        med = np.median(serie)
        mad = np.median(np.abs(serie - med))

        # most seen value
        mod = stats.mode(serie)[0][0]

        # z-score test
        zoutliers1 = (gamma * (serie - med)) / mad > beta
        zoutliers2 = (gamma * (serie - med)) / mad < -beta

        # iqr outliers test
        qoutliers1 = serie < q25 - alpha * iqr
        qoutliers2 = serie > q75 + alpha * iqr

        # test a new idea
        if bounds:
            # the bounds method instead of replacing by the median or the mode
            # will chop the series at the point where they become greater
            # that their respective test-value

            serie[zoutliers1] = ((beta * mad) / gamma) + med
            serie[zoutliers2] = (((-beta) * mad) / gamma) + med

            serie[qoutliers1] = q25 - alpha * iqr
            serie[qoutliers2] = q75 + alpha * iqr
        else:
            # with this method all values will be replaced by the same value,
            # which will be median or mode

            serie[zoutliers1] = np.nan
            serie[zoutliers2] = np.nan

            serie[qoutliers1] = np.nan
            serie[qoutliers2] = np.nan

        if method == 'mode':
            serie.fillna(mod, inplace=True)
        elif method == 'median':
            serie.fillna(med, inplace=True)

        f_df_copy.iloc[:, f_s] = serie
    return f_df_copy

################################################################################
################################################################################


# this function is everywhere, but I put here so the file is an stand-alone file
def keyvalue(f_df):
    """
    Transforms dataframe from the predictions in dataframe format
    (horizon x 45 i.e. one column of predictions per
    series) to the required format by Kaggle.
    :param f_df:
    :return:
    """
    f_df["horizon"] = range(1, f_df.shape[0] + 1)
    res = pd.melt(f_df, id_vars=["horizon"])
    res = res.rename(columns={"variable": "series"})
    res["Id"] = res.apply(lambda row: ("s" + str(row["series"].split("-")[1])
                                       + "h" + str(row["horizon"])), axis=1)
    res = res.drop(['series', 'horizon'], axis=1)
    res = res[["Id", "value"]]
    res = res.rename(columns={"value": "forecasts"})
    return res

################################################################################
################################################################################


# these two function perform a normalization over a dataframe and keep record
# of the scaler, it is thought to go forth and backwards
def scale(f_df, f_scaler):
    res_df = f_df.copy()
    res_index = res_df.index
    res_columns = res_df.columns
    rn = {}
    for f_i in range(len(res_columns)):
        rn[f_i] = res_columns[f_i]

    f_scaler.fit(res_df)
    res_df = f_scaler.transform(res_df)
    res_df = pd.DataFrame(res_df, index=res_index)
    res_df = res_df.rename(columns=rn)

    return res_df, f_scaler


def unscale(f_df, f_scaler):
    res_df = f_df.copy()
    res_index = res_df.index
    res_columns = res_df.columns
    rn = {}
    for f_i in range(len(res_columns)):
        rn[f_i] = res_columns[f_i]

    res_df = f_scaler.inverse_transform(res_df)
    res_df = pd.DataFrame(res_df, index=res_index)
    res_df = res_df.rename(columns=rn)

    return res_df

