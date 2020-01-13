from keras import Sequential
from keras.layers import Dense, LSTM
from pmdarima import auto_arima, arima

from utils import remove_outliers, normalize_series, split_sequence, smape, keyvalue, split_sequence_nn_with_past_multi_step, split_sequence_nn_with_past_outliers_multi_step
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm


def auto_arima_forecast(series, validation_series, horizon, del_outliers=False, normalize=False, plot=False):

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

    # input sequence is our data
    train_series = working_series

    # perform search for best parameters and fit
    model = auto_arima(train_series,
                       seasonal=True,
                       max_D=2,
                       m=7,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    order = model.get_params()['order']
    seasonal_order = model.get_params()['seasonal_order']

    # apparently useless model.fit(train_series)

    # perform predictions
    f_autoarima = model.predict(n_periods=horizon)

    # dataframe which contains the result
    forecast_dataframe = pd.DataFrame(index=validation_series.index)

    # if data was normalized, we need to apply the reverse transform
    if normalize:

        # first reverse log1p using expm1
        validation_forecast = f_autoarima

        # use scaler to reverse normalizing
        denormalized_forecast = scaler.inverse_transform(validation_forecast.reshape(-1, 1))
        denormalized_forecast = [val[0] for val in denormalized_forecast]

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = denormalized_forecast

    else:

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = f_autoarima

    if plot:
        plt.figure(figsize=(10, 6))

        plt.plot(series[-100:], color="blue", linestyle="-")
        plt.plot(validation_series, color="green", linestyle="-")
        plt.plot(forecast_dataframe, color="red", linestyle="--")

        plt.legend(["Train series", "Validation series", "Predicted series"])

        plt.title("Validation of auto ARIMA")

        plt.show()

    # print("SMAPE is " + str(smape(validation_series, forecasts['forecast'])))

    return smape(validation_series, forecast_dataframe['forecast']), forecast_dataframe['forecast'], order, seasonal_order


def arima_forecast(series, validation_series, horizon, order, seasonal_order, del_outliers=False, normalize=False, plot=False):

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

    # input sequence is our data
    train_series = working_series

    # perform search for best parameters and fit
    model = arima.ARIMA(order=order,
                        seasonal_order=seasonal_order,
                        suppress_warnings=True)

    model.fit(train_series)

    # perform predictions
    f_autoarima = model.predict(n_periods=horizon)

    # dataframe which contains the result
    forecast_dataframe = pd.DataFrame(index=validation_series.index)

    # if data was normalized, we need to apply the reverse transform
    if normalize:

        # first reverse log1p using expm1
        validation_forecast = f_autoarima

        # use scaler to reverse normalizing
        denormalized_forecast = scaler.inverse_transform(validation_forecast.reshape(-1, 1))
        denormalized_forecast = [val[0] for val in denormalized_forecast]

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = denormalized_forecast

    else:

        # save the forecast in the dataframe
        forecast_dataframe['forecast'] = f_autoarima

    if plot:
        plt.figure(figsize=(10, 6))

        plt.plot(series[-100:], color="blue", linestyle="-")
        plt.plot(validation_series, color="green", linestyle="-")
        plt.plot(forecast_dataframe, color="red", linestyle="--")

        plt.legend(["Train series", "Validation series", "Predicted series"])

        plt.title("Validation of auto ARIMA")

        plt.show()

    # print("SMAPE is " + str(smape(validation_series, forecasts['forecast'])))

    return smape(validation_series, forecast_dataframe['forecast']), forecast_dataframe['forecast']
