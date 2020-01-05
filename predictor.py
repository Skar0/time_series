import pandas as pd
import numpy as np
import fbprophet
from fbprophet import Prophet
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler

data_path = "data/train.csv"


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


def remove_outliers(series):
    """
    Removes outliers from the series and replaces their value by the median.
    This modifies the series.
    :param series:
    :return:
    """
    med = np.median(series)
    q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    upper = q75 + cut_off
    outliers = series > upper
    series[outliers] = np.nan
    series.fillna(med, inplace=True)
    return series


# naive seasonal => explainations ? taking previous values ?
# f_naive = [series_train[-(7 + h - 1)] for h in range(1, HORIZON + 1)]
# forecasts_naive[iseries] = f_naive


def train_and_predict_auto_arima(data_path, output_path):
    """
    Performs training and prediction of one ARIMA model per series.
    :param data_path:
    :param output_path:
    :return:
    """

    # with this dataset, the index column is day
    data = pd.read_csv(data_path, index_col="Day")

    # changing index to datetime object year - month - day
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")

    # note sure what for
    data = data.asfreq('d')

    nbr_series = len(data.columns)
    nbr_samples = data["series-1"].count()

    start_date = data.index[0]
    end_date = data.index[781]

    print("Start date "+str(start_date))
    print("End date "+str(end_date))

    interval_train = pd.date_range(start=start_date, end=end_date)
    interval_test = pd.date_range(start='2017-08-21', end='2017-09-10')

    # number of samples we are predicting
    horizon = len(interval_test)

    # separating data into train and validation set
    data_train = data.loc[interval_train]

    # dataframe which contains the result
    forecasts_autoarima = pd.DataFrame(index=interval_test)

    for serie in data_train.columns:
        print(serie)

        # select column for training
        series_train = data_train[serie]

        # keeping last year of data
        series_train = series_train.loc["2016-06-30":end_date]

        print("Train series number of samples "+str(series_train.count()))

        # remove outliers from the series
        series_train = remove_outliers(series_train)

        # scale data
        scaler = MinMaxScaler()

        # prepare data for normalization, i.e. put it in a matrix
        values = series_train.values
        values = values.reshape((len(values), 1))

        # fit and apply transformation
        normalized = scaler.fit_transform(values)

        # transform the scaled values as an array
        normalized = [val[0] for val in normalized]

        # create a series from the scaled values with correct index
        normalized_series = pd.Series(normalized, index=series_train.index)

        '''
        from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

        dr = pd.date_range(start='2015-07-01', end='2017-08-20')
        df = pd.DataFrame()
        df['date'] = dr

        cal = calendar()
        #holidays = cal.holidays(start=normalized_series.index.min(), end=normalized_series.index.max())
        #df['holiday'] = df['date'].isin(holidays)
        df['day_of_week'] = df['date'].dt.day_name()
        df.index = df['date']
        df.drop('date', 1)
        '''
        df = pd.DataFrame(index = normalized_series.index)
        df['weekday'] = df.index.weekday

        # perform search for best parameters and fit
        model = auto_arima(normalized_series, exogenous=df,start_p=0, start_q=0,
                           max_p=0, max_q=0, m=7,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
        model.fit(normalized_series)

        # perform predictions
        f_autoarima = model.predict(n_periods=horizon)

        # reshape the array of predicted values and apply inverse transform
        inversed = scaler.inverse_transform(f_autoarima.reshape(-1, 1))

        # the an array of inversed transformed values
        inversed = [val[0] for val in inversed]

        # save the values
        forecasts_autoarima[serie] = inversed

    # transform the predictions into correct format
    forecasts_autoarima = keyvalue(forecasts_autoarima)
    # write output
    forecasts_autoarima.to_csv(output_path, index=False)


def train_and_predict_prophet(data_path, output_path):
    """
    Performs training and prediction of one facebook prophet model per series.
    :param data_path:
    :param output_path:
    :return:
    """

    # with this dataset, the index column is day
    data = pd.read_csv(data_path, index_col="Day")

    # changing index to datetime object year - month - day
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")

    # note sure what for
    data = data.asfreq('d')

    nbr_series = len(data.columns)
    nbr_samples = data["series-1"].count()

    start_date = data.index[0]
    end_date = data.index[781]

    # for this instance, and because of prphet library requirements we use all available data for training
    interval_train = pd.date_range(start=start_date, end=end_date)
    interval_valid = pd.date_range(start='2017-07-1', end=end_date)
    interval_test = pd.date_range(start='2017-08-21', end='2017-09-10')

    # number of samples we are predicting
    horizon = len(interval_test)

    # separating data into train and validation set
    data_train = data.loc[interval_train]
    data_valid = data.loc[interval_valid]

    # dataframe which contains the result
    forecasts_prophet = pd.DataFrame(index=interval_test)

    for serie in data_train.columns:
        print(serie)

        # select column for training
        series_train = data_train[serie]

        # remove outliers
        series_train = remove_outliers(series_train)

        # format the data for prophet library
        frame = {'ds': series_train.index, 'y': series_train}
        formated = pd.DataFrame(frame)

        # create prophet object and fit data
        prophet = Prophet()
        prophet.fit(formated)

        # make dataframe for the future (this is the list of dates of training + the list of dates we must predict)
        future = prophet.make_future_dataframe(periods=horizon)

        # perform forecasting
        forecast = prophet.predict(future)
        # fig = prophet.plot(forecast)

        # not sure why, but removing test below produces nan values in final result
        test = forecast["yhat"]
        test.index = forecast.ds
        test = test.loc['2017-08-21': '2017-09-10']

        # yhat is the predicted value, assign it as our prediction
        forecasts_prophet[serie] = forecast["yhat"]

    # transform the predictions into correct format
    forecasts_prophet = keyvalue(forecasts_prophet)
    # write output
    forecasts_prophet.to_csv(output_path, index=False)


train_and_predict_auto_arima("data/train.csv", "data/sarima_scaled")
