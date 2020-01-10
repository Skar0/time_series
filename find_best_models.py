import pandas as pd
from matplotlib import pyplot as plt
from nn_multi_step_forecasting import arima_forecast, nn_multi_step_forecast, nn_multi_step_forecast_mse, \
    nn_with_past_multi_step_forecast, nn_with_past_outliers_multi_step_forecast
from nn_single_step_forecasting import nn_single_step_forecast, nn_with_past_single_step_forecast, \
    nn_with_past_outliers_single_step_forecast
from utils import keyvalue


def evaluate_best_model():
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

    # training set without validation for the submission
    interval_train_full = pd.date_range(start=start_date, end=end_date)

    # validation is 21 days
    interval_valid = pd.date_range(start='2017-07-31', end=end_date)

    # test is 21 days
    interval_test = pd.date_range(start='2017-08-21', end='2017-09-10')

    # number of samples we are predicting
    horizon = len(interval_test)

    # separating data into train and validation set
    data_train = data.loc[interval_train]
    data_train_full = data.loc[interval_train_full]
    data_valid = data.loc[interval_valid]
    data_test = pd.DataFrame(index=interval_test)
    data_submit = pd.DataFrame(index=interval_test)

    #nn_forecast_params = [7, 25, 5, 5, 7, 30, 7, 20, 10, 25, 7, 7, 5, 25, 7, 5, 30, 5, 5, 5, 14, 5, 25, 14, 14, 10, 7,
    #                      5, 7, 14, 10, 5, 14, 10, 14, 30, 30, 14, 14, 7, 25, 25, 5, 7, 14]

    forecast_in = [5, 7, 14, 25]
    methods = [arima_forecast, nn_multi_step_forecast, nn_multi_step_forecast_mse, nn_with_past_multi_step_forecast, nn_with_past_outliers_multi_step_forecast,
               nn_single_step_forecast, nn_with_past_single_step_forecast, nn_with_past_outliers_single_step_forecast]
    methods_colors = ["blue", "red", "orange", "brown", "pink", "magenta", "grey", "yellow"]
    methods_smapes = {}

    methods_smapes[arima_forecast.__name__] = []
    for method in methods[1:]:
        for s in forecast_in:
            methods_smapes[method.__name__+"-"+str(s)] = []
    chosen_method = []
    chosen_method_smape = []
    chosen_method_param = []

    for i in range(1, nbr_series + 1):

        best_smape = 1000000000
        best_model = None
        best_model_param = None

        plt.figure(figsize=(10, 6))

        plt.plot(data_train['series-' + str(i)][-65:], color="black", linestyle="-")
        plt.plot(data_valid['series-' + str(i)], color="green", linestyle="-")


        """ Special plotting for ARIMA """

        print("Current method " + arima_forecast.__name__)

        arima_smape, arima_series = arima_forecast(data_train['series-' + str(i)], data_valid['series-' + str(i)],
                                                   (1, 1, 2), (0, 1, 1, 7), horizon, del_outliers=True, normalize=True,
                                                   plot=False)

        methods_smapes[arima_forecast.__name__].append(arima_smape)

        plt.plot(arima_series, color=methods_colors[0], linestyle="--")

        if arima_smape < best_smape :
            best_smape = arima_smape
            best_model = arima_forecast

        used_color = 1
        """ Plot for the NN"""
        for method in methods[1:]:
            for size in forecast_in:
                print("Current method "+method.__name__+"-"+str(size))

                nn_smape, nn_series = method(data_train['series-' + str(i)], data_valid['series-' + str(i)],
                                                            size, horizon, del_outliers=True, normalize=True,
                                                             plot=False)
                methods_smapes[method.__name__+"-"+str(size)].append(nn_smape)
                plt.plot(nn_series, color=methods_colors[used_color], linestyle="--")

                if nn_smape < best_smape:
                    best_smape = nn_smape
                    best_model = method
                    best_model_param = size
        used_color+=1

        names = ["Train series", "Validation series"]
        for name in methods:
            names.append(name.__name__)
        plt.legend(names)
        plt.title("Series " + str(i) + "models comparison")
        plt.savefig('data/series-' + str(i) + "-models-comparison.pdf")
        plt.show()

        print()
        for key in methods_smapes.keys():
            print("SERIE " + str(i) + " SMAPE for "+key.to_string()+": " + str(methods_smapes[key][i-1]))
        print()

        plt.figure(figsize=(10, 6))
        plt.plot(data_train_full['series-' + str(i)][-65:], color="blue", linestyle="-")

        # check best model
        if best_model == arima_forecast:
            submit_smape, submit_series = arima_forecast(data_train_full['series-' + str(i)], data_test,
                                                         (1, 1, 2), (0, 1, 1, 7), horizon, del_outliers=True,
                                                         normalize=True,
                                                         plot=False)
            chosen_method.append(arima_forecast.__name__)
            chosen_method_smape.append(arima_smape)
            chosen_method_param.append(((1, 1, 2), (0, 1, 1, 7)))
        else:
            submit_smape, submit_series = best_model(data_train_full['series-' + str(i)], data_test,
                                                            best_model_param, horizon, del_outliers=True, normalize=True,
                                                             plot=False)
            chosen_method.append(best_model.__name__)
            chosen_method_smape.append(methods_smapes[best_model.__name__+"-"+str(best_model_param)][i-1])
            chosen_method_param.append(best_model_param)

        data_submit['series-' + str(i)] = submit_series

        plt.plot(submit_series, color="red", linestyle="--")

        plt.legend(["Train series", "Forecasting using " + str(chosen_method[i - 1])])
        plt.title("Series " + str(i) + " forecasting")
        plt.savefig('data/series-' + str(i) + "-forecasting.pdf")
        plt.show()

        print("-" * 100)

    print(data_submit.to_string())
    print(chosen_method)
    print(chosen_method_smape)

    print()
    submission = keyvalue(data_submit)
    print(submission.to_string())
    submission.to_csv("all_vs.csv")


evaluate_best_model()