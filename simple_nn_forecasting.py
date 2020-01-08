from keras import Sequential
from keras.layers import Dense
from tools import remove_outliers, normalize_series, split_sequence, smape, keyvalue
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def nn_horizon_forecast(series, validation_series, horizon, del_outliers=False, normalize=False, plot=False):

    if del_outliers:
        working_series = remove_outliers(series)

    else:
        working_series = series

    if normalize:
        scaler, working_series = normalize_series(working_series)

    # input sequence is our data
    raw_seq = working_series
    raw_seq = np.log1p(working_series)

    # we use the last 10 days as input and predict the full horizon
    n_steps_in, n_steps_out = 10, horizon

    # split into samples
    X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

    # create the model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=n_steps_in))
    model.add(Dense(n_steps_out))

    # we use 'mae' with data transformed with log1p and expm1 as it approaches smape values
    model.compile(optimizer='adam', loss='mae')

    # fit model
    model.fit(X, y, epochs=200, verbose=0)

    # perform prediction

    # input is the last n_steps_in values of the train series since we predict the next horizon days after that
    x_input = np.log1p(np.array(working_series.values[-n_steps_in:]))
    x_input = x_input.reshape((1, n_steps_in))
    nn_forecast = model.predict(x_input, verbose=0)

    # dataframe which contains the result
    forecasts = pd.DataFrame(index=validation_series.index)

    if normalize:
        forecast = np.expm1(nn_forecast)

        inversed = scaler.inverse_transform(forecast.reshape(-1, 1))

        inversed = [val[0] for val in inversed]

        forecasts['forecast'] = inversed

    else:
        forecasts['forecast'] = np.expm1(nn_forecast)

    if plot:
        plt.figure(figsize=(10, 6))

        plt.plot(series[-100:], color="blue", linestyle="-")
        #plt.plot(validation_series, color="green", linestyle="-")
        plt.plot(forecasts, color="red", linestyle="--")

        plt.legend(["Train series", "Validation series", "Predicted series"])

        plt.title("Validation of simple NN with input size "+str(n_steps_in)+" output size "+str(n_steps_out))

        plt.show()

    print("SMAPE is " + str(smape(validation_series, forecasts['forecast'])))

    return smape(validation_series, forecasts['forecast']), forecasts['forecast']


def evaluate_nn_horizon_forecast():

    # with this dataset, the index column is day
    data = pd.read_csv("data/train.csv", index_col="Day")

    # changing index to datetime object year - month - day
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
    data = data.asfreq('d')

    nbr_series = len(data.columns)
    nbr_samples = data["series-1"].count()

    start_date = data.index[0]
    end_date = data.index[781]

    print("Start date " + str(start_date))
    print("End date " + str(end_date))

    interval_train = pd.date_range(start=start_date, end='2017-07-30')

    # validation is 21 days
    interval_valid = pd.date_range(start='2017-07-31', end=end_date)

    # test is 21 days
    interval_test = pd.date_range(start='2017-08-21', end='2017-09-10')

    # number of samples we are predicting
    horizon = len(interval_test)

    # separating data into train and validation set
    data_train = data.loc[interval_train]
    data_valid = data.loc[interval_valid]

    scores = {}
    # for each series, forecast and record scores
    for i in range(1, nbr_series + 1):

        smape_val, forecast = nn_horizon_forecast(data_train['series-'+str(i)], data_valid['series-'+str(i)], horizon,
                       del_outliers=True, normalize=True, plot=False)

        scores[i] = smape_val

        print("series" + str(i) + " smape "+str(smape_val))

    print(scores)


def submit_nn_horizon_forecast():

    # with this dataset, the index column is day
    data = pd.read_csv("data/train.csv", index_col="Day")

    # changing index to datetime object year - month - day
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
    data = data.asfreq('d')

    nbr_series = len(data.columns)
    nbr_samples = data["series-1"].count()

    start_date = data.index[0]
    end_date = data.index[781]

    print("Start date " + str(start_date))
    print("End date " + str(end_date))

    # here training set is full dataset
    interval_train = pd.date_range(start=start_date, end=end_date)

    # test is 21 days
    interval_test = pd.date_range(start='2017-08-21', end='2017-09-10')

    # number of samples we are predicting
    horizon = len(interval_test)

    # separating data into train and validation set
    data_train = data.loc[interval_train]

    data_test = pd.DataFrame(index=interval_test)

    data_submit = pd.DataFrame(index=interval_test)

    scores = {}
    # for each series, forecast and record scores
    for i in range(1, nbr_series + 1):

        smape_val, forecast = nn_horizon_forecast(data_train['series-'+str(i)], data_test, horizon,
                       del_outliers=True, normalize=True, plot=False)

        scores[i] = smape_val

        data_submit['series-'+str(i)] = forecast

        print(data_submit.to_string())
        print("series" + str(i) + " smape "+str(smape_val))

    print(scores)

    submission = keyvalue(data_submit)
    submission.to_csv("submission.csv")


evaluate_nn_horizon_forecast()