import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

from utils import remove_outliers, normalize_series, split_sequence, smape, split_sequence_nn_with_past_multi_step, \
    split_sequence_nn_with_past_outliers_multi_step


def nn_single_step_forecast(series, validation_series, input_length, horizon, del_outliers=False, normalize=False,
                            plot=False):
    """
    Perform forecasting of a time series using a simple neural network with a single 128 neurons hidden layer.
    The network is trained using samples of shape input_length (corresponding to the last input_length days) to predict
    an array of horizon values (corresponding to horizon days). In this case, the network predicts one day at the time.
    Performance of the trained network is assessed on a validation series. This is computed by repeating one day
    predictions and shifting the input values. The size of the validation series must be horizon.

    :param series:
    :param validation_series:
    :param input_length:
    :param horizon:
    :param del_outliers:
    :param normalize:
    :param plot:
    :return: SMAPE for the validation series, the forecast validation series
    """

    # whether to remove outliers in the training series
    if del_outliers:
        working_series = remove_outliers(series)

    else:
        working_series = series

    # whether to normalize the training series
    if normalize:
        scaler, working_series = normalize_series(working_series)

    else:
        scaler = None

    # input sequence is our data, np.log1p is applied to the data and mae error is used to approximate SMAPE error
    train_series = np.log1p(working_series)

    # we use the last n_steps_in days as input and predict one step
    n_steps_in, n_steps_out = input_length, 1

    # split into samples
    train_samples, train_targets = split_sequence(train_series, n_steps_in, n_steps_out)

    # create the model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=n_steps_in))

    # we predict n_steps_out values
    model.add(Dense(n_steps_out))

    # we use 'mae' with data transformed with log1p and expm1 to approach SMAPE error
    model.compile(optimizer='adam', loss='mae')

    # fit model
    model.fit(train_samples, train_targets, epochs=200, verbose=0)

    # perform prediction

    # we start by transforming the normalized series into log1p, new one day predictions will be added to this series
    # as we predict them and these predictions will be used for the next forecasting step
    working_series_values = np.log1p(working_series.values)

    # perform horizon predictions
    for i in range(horizon):
        validation_in_sample = np.array(working_series_values[-n_steps_in:])
        validation_in_sample = validation_in_sample.reshape((1, n_steps_in))

        validation_forecast = model.predict(validation_in_sample, verbose=0)

        working_series_values = np.append(working_series_values, validation_forecast)

    # take last horizon values from the series (this is the forecast for the validation series
    validation_forecast = working_series_values[-horizon:]

    # dataframe which contains the result
    forecast_dataframe = pd.DataFrame(index=validation_series.index)

    # if data was normalized, we need to apply the reverse transform
    if normalize:

        # first reverse log1p using expm1
        validation_forecast = np.expm1(validation_forecast)

        # use scaler to reverse normalizing
        denormalized_forecast = scaler.inverse_transform(validation_forecast.reshape(-1, 1))
        denormalized_forecast = [val[0] for val in denormalized_forecast]

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = denormalized_forecast

    else:

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = np.expm1(validation_forecast)

    if plot:
        plt.figure(figsize=(10, 6))

        plt.plot(series[-100:], color="blue", linestyle="-")
        plt.plot(validation_series, color="green", linestyle="-")
        plt.plot(forecast_dataframe, color="red", linestyle="--")

        plt.legend(["Train series", "Validation series", "Predicted series"])

        plt.title(
            "Validation of single step NN with input size " + str(n_steps_in) + " output size " + str(n_steps_out))

        plt.show()

    return smape(validation_series, forecast_dataframe['forecast']), forecast_dataframe['forecast']


def nn_with_past_single_step_forecast(series, validation_series, input_length, horizon, del_outliers=False,
                                      normalize=False, plot=False):
    """
    Perform forecasting of a time series using a simple neural network with a single 128 neurons hidden layer.
    The network is trained using samples of shape input_length (corresponding to the last input_length days) to predict
    an array of horizon values (corresponding to horizon days). In this case, the network predicts one day at the time.
    Performance of the trained network is assessed on a validation series. This is computed by repeating one day
    predictions and shifting the input values. The size of the validation series must be horizon.

    This function differs from nn_single_step_forecast as in addition to the last input_length days, we also use the
    value from the same day the previous year as an input to the network. The hope is to gain information from the
    previous year.

    :param series:
    :param validation_series:
    :param input_length:
    :param horizon:
    :param del_outliers:
    :param normalize:
    :param plot:
    :return: SMAPE for the validation series, the forecast validation series
    """

    # whether to remove outliers in the training series
    if del_outliers:
        working_series = remove_outliers(series)

    else:
        working_series = series

    # whether to normalize the training series
    if normalize:
        scaler, working_series = normalize_series(working_series)

    else:
        scaler = None

    # input sequence is our data, np.log1p is applied to the data and mae error is used to approximate SMAPE error
    train_series = np.log1p(working_series)

    # we use the last n_steps_in days as input and predict one step
    n_steps_in, n_steps_out = input_length, 1

    # split into samples, using sample from previous year
    # implementation from multi steps can be used here since single step is special case of multi steps
    train_samples, train_targets = split_sequence_nn_with_past_multi_step(train_series, n_steps_in, n_steps_out)

    # create the model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=n_steps_in + 1))

    # we predict n_steps_out values
    model.add(Dense(n_steps_out))

    # we use 'mae' with data transformed with log1p and expm1 to approach SMAPE error
    model.compile(optimizer='adam', loss='mae')

    # fit model
    model.fit(train_samples, train_targets, epochs=200, verbose=0)

    # perform prediction

    # we start by transforming the normalized series into log1p, new one day predictions will be added to this series
    # as we predict them and these predictions will be used for the next forecasting step
    working_series_values = np.log1p(working_series.values)

    # perform horizon predictions
    for i in range(horizon):
        # input contains the value from the previous year for the forecast day
        validation_in_sample = np.append(np.array(working_series_values[-365 + 1]),
                                         np.array(working_series_values[-n_steps_in:]))
        validation_in_sample = validation_in_sample.reshape((1, n_steps_in + 1))

        validation_forecast = model.predict(validation_in_sample, verbose=0)

        working_series_values = np.append(working_series_values, validation_forecast)

    # take last horizon values from the series (this is the forecast for the validation series
    validation_forecast = working_series_values[-horizon:]

    # dataframe which contains the result
    forecast_dataframe = pd.DataFrame(index=validation_series.index)

    # if data was normalized, we need to apply the reverse transform
    if normalize:

        # first reverse log1p using expm1
        validation_forecast = np.expm1(validation_forecast)

        # use scaler to reverse normalizing
        denormalized_forecast = scaler.inverse_transform(validation_forecast.reshape(-1, 1))
        denormalized_forecast = [val[0] for val in denormalized_forecast]

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = denormalized_forecast

    else:

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = np.expm1(validation_forecast)

    if plot:
        plt.figure(figsize=(10, 6))

        plt.plot(series[-100:], color="blue", linestyle="-")
        plt.plot(validation_series, color="green", linestyle="-")
        plt.plot(forecast_dataframe, color="red", linestyle="--")

        plt.legend(["Train series", "Validation series", "Predicted series"])

        plt.title(
            "Validation of single step NN with input size " + str(n_steps_in) + " output size " + str(n_steps_out))

        plt.show()

    return smape(validation_series, forecast_dataframe['forecast']), forecast_dataframe['forecast']


def nn_with_past_outliers_single_step_forecast(series, validation_series, input_length, horizon, del_outliers=False,
                                               normalize=False, plot=False):
    """
    Perform forecasting of a time series using a simple neural network with a single 128 neurons hidden layer.
    The network is trained using samples of shape input_length (corresponding to the last input_length days) to predict
    an array of horizon values (corresponding to horizon days). In this case, the network predicts one day at the time.
    Performance of the trained network is assessed on a validation series. This is computed by repeating one day
    predictions and shifting the input values. The size of the validation series must be horizon.

    This function differs from nn_single_step_forecast as in addition to the last input_length days, we also use the
    value from the same day the previous year as an input to the network. This value is normalized but contains the
    outliers. The hope is to gain information from the previous year.

    :param series:
    :param validation_series:
    :param input_length:
    :param horizon:
    :param del_outliers:
    :param normalize:
    :param plot:
    :return: SMAPE for the validation series, the forecast validation series
    """

    # whether to remove outliers in the training series
    if del_outliers:
        working_series = remove_outliers(series)

    else:
        working_series = series

    # whether to normalize the training series
    if normalize:
        scaler, working_series = normalize_series(working_series)
        scaler_bis, working_series_with_outliers = normalize_series(series)
    else:
        scaler = None
        working_series_with_outliers = series

    # input sequence is our data, np.log1p is applied to the data and mae error is used to approximate SMAPE error
    train_series = np.log1p(working_series)

    # we use the last n_steps_in days as input and predict one step
    n_steps_in, n_steps_out = input_length, 1

    # split into samples, using sample from previous year
    # implementation from multi steps can be used here since single step is special case of multi steps
    train_samples, train_targets = split_sequence_nn_with_past_outliers_multi_step(train_series,
                                                                                   working_series_with_outliers,
                                                                                   n_steps_in, n_steps_out)

    # create the model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=n_steps_in + 1))

    # we predict n_steps_out values
    model.add(Dense(n_steps_out))

    # we use 'mae' with data transformed with log1p and expm1 to approach SMAPE error
    model.compile(optimizer='adam', loss='mae')

    # fit model
    model.fit(train_samples, train_targets, epochs=200, verbose=0)

    # perform prediction

    # we start by transforming the normalized series into log1p, new one day predictions will be added to this series
    # as we predict them and these predictions will be used for the next forecasting step
    working_series_values = np.log1p(working_series.values)

    # perform horizon predictions
    for i in range(horizon):
        # input contains the value from the previous year for the forecast day
        validation_in_sample = np.append(np.array(working_series_with_outliers[-365 + 1]),
                                         np.array(working_series_values[-n_steps_in:]))
        validation_in_sample = validation_in_sample.reshape((1, n_steps_in + 1))

        validation_forecast = model.predict(validation_in_sample, verbose=0)

        working_series_values = np.append(working_series_values, validation_forecast)

        working_series_with_outliers = np.append(working_series_with_outliers, validation_forecast)

    # take last horizon values from the series (this is the forecast for the validation series
    validation_forecast = working_series_values[-horizon:]

    # dataframe which contains the result
    forecast_dataframe = pd.DataFrame(index=validation_series.index)

    # if data was normalized, we need to apply the reverse transform
    if normalize:

        # first reverse log1p using expm1
        validation_forecast = np.expm1(validation_forecast)

        # use scaler to reverse normalizing
        denormalized_forecast = scaler.inverse_transform(validation_forecast.reshape(-1, 1))
        denormalized_forecast = [val[0] for val in denormalized_forecast]

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = denormalized_forecast

    else:

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = np.expm1(validation_forecast)

    if plot:
        plt.figure(figsize=(10, 6))

        plt.plot(series[-100:], color="blue", linestyle="-")
        plt.plot(validation_series, color="green", linestyle="-")
        plt.plot(forecast_dataframe, color="red", linestyle="--")

        plt.legend(["Train series", "Validation series", "Predicted series"])

        plt.title("Validation of simple NN with input size " + str(n_steps_in) + " output size " + str(n_steps_out))

        plt.show()

    return smape(validation_series, forecast_dataframe['forecast']), forecast_dataframe['forecast']
