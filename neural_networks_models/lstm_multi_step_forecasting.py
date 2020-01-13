import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot as plt
from utils import remove_outliers, normalize_series, split_sequence, smape


def stacked_lstm_multi_step_forecast(series, validation_series, input_length, horizon, del_outliers=False,
                                     normalize=False, plot=False):
    """
    Perform forecasting of a time series using an lstm neural network. The network is trained using samples of shape
    input_length (corresponding to the last input_length days) to predict an array of horizon values (corresponding to
    horizon days). In this case, the network predicts horizon days at the time. Performance of the trained network is
    assessed on a validation series. The size of the validation series must be horizon.

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

    # we use the last n_steps_in days as input and predict n_steps_out
    n_steps_in, n_steps_out = input_length, horizon

    # split into samples
    train_samples, train_targets = split_sequence(train_series, n_steps_in, n_steps_out)

    # here we work with the original series so only the actual values
    n_features = 1
    train_samples = train_samples.reshape((train_samples.shape[0], train_samples.shape[1], n_features))

    # create the model
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(n_steps_in, n_features)))

    # we predict n_steps_out values
    model.add(Dense(n_steps_out))

    # we use 'mae' with data transformed with log1p and expm1 to approach SMAPE error
    model.compile(optimizer='adam', loss='mae')

    # fit model
    model.fit(train_samples, train_targets, epochs=200, verbose=0)

    # perform prediction

    # input is the last n_steps_in values of the train series (working_series is not log1p transformed)
    validation_in_sample = np.log1p(np.array(working_series.values[-n_steps_in:]))
    validation_in_sample = validation_in_sample.reshape((1, n_steps_in, n_features))
    validation_forecast = model.predict(validation_in_sample, verbose=0)

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

        plt.title("Validation of LSTM with input size " + str(n_steps_in) + " output size " + str(n_steps_out))

        plt.show()

    return smape(validation_series, forecast_dataframe['forecast']), forecast_dataframe['forecast']
