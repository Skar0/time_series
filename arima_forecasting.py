import pandas as pd
from matplotlib import pyplot as plt
from pmdarima import auto_arima, arima

from utils import remove_outliers, normalize_series, smape


def auto_arima_forecast(series, validation_series, horizon, del_outliers=False, normalize=False, plot=False):
    """
    Fits an auto arima model from the series to find the best parameters. Performance of the trained model is assessed
    on a validation series.

    :param series:
    :param validation_series:
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

        plt.title("Validation of auto arima model")

        plt.show()

    return smape(validation_series, forecast_dataframe['forecast']), forecast_dataframe['forecast'], order, seasonal_order


def arima_forecast(series, validation_series, horizon, order, seasonal_order, del_outliers=False, normalize=False, plot=False):
    """
    Creates an arima model with the provided order and seasonal order and assess performance of the model is on a
    validation series.

    :param series:
    :param validation_series:
    :param horizon:
    :param order:
    :param seasonal_order:
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

        plt.title("Validation of arima model with order "+str(order)+" seasonal order "+str(seasonal_order))

        plt.show()

    return smape(validation_series, forecast_dataframe['forecast']), forecast_dataframe['forecast']
