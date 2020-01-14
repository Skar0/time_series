import copy

import pandas as pd
from matplotlib import pyplot as plt
from neural_networks_models.nn_single_step_forecasting import nn_single_step_forecast, nn_with_past_outliers_single_step_forecast, \
    nn_with_past_single_step_forecast

from arima_forecasting import auto_arima_forecast, arima_forecast
from neural_networks_models.nn_multi_step_forecasting import nn_multi_step_forecast, nn_with_past_multi_step_forecast, \
    nn_with_past_outliers_multi_step_forecast
# These models and parameters have been chosen through experiments
from utils import remove_outliers, keyvalue

ARIMA_parameters = [
[(1, 1, 2), (1, 0, 1, 7)],
[(1, 1, 3), (1, 0, 0, 7)],
[(4, 1, 1), (2, 0, 2, 7)],
[(1, 1, 2), (0, 0, 0, 7)],
[(1, 1, 1), (2, 0, 2, 7)],
[(1, 1, 1), (2, 0, 2, 7)],
[(1, 0, 3), (1, 0, 1, 7)],
[(1, 1, 2), (2, 0, 1, 7)],
[(2, 1, 1), (1, 0, 2, 7)],
[(3, 1, 1), (2, 0, 2, 7)],
[(1, 1, 2), (1, 0, 1, 7)],
[(1, 1, 1), (0, 0, 0, 7)],
[(3, 1, 1), (2, 0, 2, 7)],
[(1, 1, 1), (1, 0, 1, 7)],
[(1, 1, 2), (1, 0, 1, 7)],
[(2, 0, 2), (1, 0, 1, 7)],
[(2, 1, 1), (1, 0, 1, 7)],
[(1, 1, 2), (2, 0, 0, 7)],
[(2, 1, 2), (1, 0, 1, 7)],
[(2, 1, 2), (2, 0, 2, 7)],
[(1, 0, 2), (1, 0, 1, 7)],
[(1, 1, 1), (1, 0, 1, 7)],
[(1, 1, 1), (0, 0, 0, 7)],
[(1, 1, 2), (1, 0, 1, 7)],
[(2, 1, 2), (1, 0, 1, 7)],
[(1, 1, 2), (2, 0, 2, 7)],
[(2, 1, 1), (1, 0, 1, 7)],
[(1, 1, 1), (1, 0, 0, 7)],
[(1, 1, 2), (2, 0, 2, 7)],
[(1, 1, 2), (2, 0, 2, 7)],
[(1, 1, 1), (1, 0, 0, 7)],
[(1, 0, 2), (1, 0, 1, 7)],
[(1, 1, 1), (2, 0, 0, 7)],
[(0, 1, 2), (2, 0, 2, 7)],
[(0, 1, 2), (1, 0, 1, 7)],
[(2, 1, 1), (1, 0, 1, 7)],
[(1, 1, 1), (0, 0, 1, 7)],
[(1, 1, 2), (2, 0, 2, 7)],
[(0, 1, 2), (2, 0, 2, 7)],
[(2, 1, 1), (2, 0, 2, 7)],
[(1, 1, 1), (2, 0, 2, 7)],
[(1, 1, 1), (1, 0, 0, 7)],
[(2, 1, 1), (1, 0, 0, 7)],
[(0, 1, 3), (0, 0, 1, 7)],
[(1, 1, 2), (1, 0, 1, 7)]
]


models_to_try_parameters = [
    [5, 30, 25],
    [25, 14, 14],
    [14, 25, 25],
    [5, 14, 7],
    [5, 5, 7],
    [14, 5, 14],
    [7, 7, 7],
    [14, 14, 7],
    [7, 7],
    [7, 14, 25],
    [7, 7],
    [7, 5, 5],
    [5, 7, 25],
    [7, 25, 25],
    [25, 5, 5],
    [14, 14, 14],
    [25, 7, 14],
    [7, 7, 5, 5],
    [5, 5, 5, 5],
    [5, 7, 7, 7],
    [14, 7, 7, 14],
    [25, 5, 14, 25],
    [14, 7, 25],
    [14, 5, 14],
    [14, 7, 14],
    [7, 7, 7],
    [7, 14, 7, 5],
    [14, 14, 7],
    [5, 5, 7],
    [14, 5, 5],
    [7, 5, 14],
    [5, 5, 7],
    [25, 7, 7],
    [14, 7, 7, 7],
    [7, 5, 25],
    [7, 5, 7],
    [25, 25, 14],
    [5, 5, 14],
    [14, 7, 7, 14],
    [14, 14, 14, 14],
    [14, 14, 14, 14],
    [14, 14, 5],
    [7, 14, 7],
    [7, 5, 5, 7],
    [7, 25, 7]
]
models_to_try = [
        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_outliers_single_step_forecast],

        [nn_single_step_forecast, nn_with_past_outliers_single_step_forecast,
         nn_with_past_single_step_forecast],

        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_multi_step_forecast],

        [nn_single_step_forecast, nn_with_past_single_step_forecast, nn_with_past_multi_step_forecast],

        [nn_with_past_single_step_forecast, nn_with_past_multi_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_multi_step_forecast,
         nn_with_past_outliers_multi_step_forecast],

        [nn_with_past_outliers_single_step_forecast, nn_multi_step_forecast,
         nn_with_past_multi_step_forecast],

        [nn_multi_step_forecast, nn_with_past_multi_step_forecast],

        [nn_multi_step_forecast, nn_with_past_single_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_single_step_forecast],

        [nn_with_past_single_step_forecast, nn_single_step_forecast, nn_multi_step_forecast],

        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_outliers_single_step_forecast],

        [nn_with_past_outliers_multi_step_forecast, nn_multi_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_with_past_multi_step_forecast, nn_with_past_single_step_forecast, nn_multi_step_forecast],

        [nn_multi_step_forecast, nn_with_past_multi_step_forecast, nn_single_step_forecast],

        [nn_with_past_multi_step_forecast, nn_with_past_outliers_single_step_forecast,
         nn_single_step_forecast],

        [nn_single_step_forecast, nn_with_past_outliers_single_step_forecast,
         nn_with_past_single_step_forecast, nn_multi_step_forecast],

        [nn_single_step_forecast, nn_multi_step_forecast, nn_with_past_single_step_forecast,
         nn_with_past_outliers_multi_step_forecast],

        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_single_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_with_past_single_step_forecast, nn_with_past_outliers_single_step_forecast,
         nn_single_step_forecast, nn_multi_step_forecast],

        [nn_single_step_forecast, nn_with_past_multi_step_forecast, nn_with_past_single_step_forecast,
         nn_multi_step_forecast],

        [nn_with_past_outliers_multi_step_forecast, nn_single_step_forecast, nn_multi_step_forecast],

        [nn_multi_step_forecast, nn_with_past_outliers_single_step_forecast, nn_single_step_forecast],

        [nn_with_past_multi_step_forecast, nn_multi_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_outliers_multi_step_forecast, nn_single_step_forecast],

        [nn_with_past_outliers_single_step_forecast, nn_with_past_outliers_multi_step_forecast,
         nn_with_past_multi_step_forecast, nn_multi_step_forecast],

        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_single_step_forecast],

        [nn_single_step_forecast, nn_with_past_outliers_single_step_forecast, nn_multi_step_forecast],

        [nn_multi_step_forecast, nn_with_past_outliers_multi_step_forecast,
         nn_with_past_single_step_forecast],

        [nn_with_past_outliers_single_step_forecast, nn_multi_step_forecast,
         nn_with_past_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_outliers_multi_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_single_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_multi_step_forecast, nn_single_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_outliers_single_step_forecast, nn_single_step_forecast],

        [nn_single_step_forecast, nn_with_past_single_step_forecast, nn_with_past_multi_step_forecast],

        [nn_multi_step_forecast, nn_with_past_multi_step_forecast,
         nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_outliers_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_multi_step_forecast,
         nn_with_past_outliers_multi_step_forecast, nn_single_step_forecast],

        [nn_multi_step_forecast, nn_with_past_multi_step_forecast,
         nn_with_past_outliers_multi_step_forecast, nn_single_step_forecast],

        [nn_with_past_multi_step_forecast, nn_with_past_outliers_multi_step_forecast,
         nn_single_step_forecast, nn_with_past_single_step_forecast],

        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_multi_step_forecast],

        [nn_multi_step_forecast, nn_with_past_outliers_single_step_forecast,
         nn_with_past_single_step_forecast],

        [nn_multi_step_forecast, nn_single_step_forecast, nn_with_past_multi_step_forecast,
         nn_with_past_single_step_forecast],

        [nn_with_past_outliers_single_step_forecast, nn_multi_step_forecast,
         nn_with_past_single_step_forecast]
    ]

def compute_and_plot_submission(save_path, fig_folder):
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

    # for plotting
    methods_colors = ["blue", "red", "cyan", "orange", "pink", "magenta", "grey", "yellow"]

    # to store the chosen model
    chosen_method = []
    chosen_method_smape = []
    chosen_method_param = []

    # to record the smape of each model
    record_smapes = []

    number_of_days = 62

    for i in range(1, nbr_series + 1):

        # to perform comparisons on models
        best_smape = 1000000000
        best_model = None
        best_model_param = None

        nn_models_current_series = models_to_try[i-1]
        nn_models_params_current_series = models_to_try_parameters[i-1]
        nbr_models_current_series = len(nn_models_current_series)
        record_smapes.append([])

        # for validation, we plot each validation forecast separately
        # axis 0 is a comparison of each forecast on last 90 days
        # axis 1 is a comparison of each forecast on the whole series
        _, ax = plt.subplots(nrows=nbr_models_current_series + 2, ncols= 1, figsize=(12, 10))

        ax[0].set_title('Comparison of model forecast on validation series for series '+str(i)+' (last '+str(number_of_days)+' days)')
        ax[0].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
        ax[0].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        ax[1].set_title('Comparison of model forecast on validation series for series '+str(i)+' (whole series without outliers)')
        ax[1].plot(remove_outliers(data_train['series-' + str(i)]), color="black", linestyle="-")
        ax[1].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        axis_legend = ["Train series", "Validation series"]
        legend_size = 6
        for model_name in nn_models_current_series:
            axis_legend.append(model_name.__name__)
        ax[0].legend(axis_legend, prop={'size': legend_size})
        ax[1].legend(axis_legend, prop={'size': legend_size})
        axis_legend = ["Train series", "Validation series"]

        for model_index in range(nbr_models_current_series):
            ax[model_index + 2].set_title(
                'Comparison of model forecast on validation series for series ' + str(i) +' (last '+str(number_of_days)+' days)')
            ax[model_index + 2].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
            ax[model_index + 2].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

            smape, forecast = nn_models_current_series[model_index](data_train['series-' + str(i)],
                                                                    data_valid['series-' + str(i)],
                                                                    nn_models_params_current_series[model_index],
                                                                    horizon,
                                                                    del_outliers=True,
                                                                    normalize=True,
                                                                    plot=False)

            ax[0].plot(forecast, color=methods_colors[model_index], linestyle="--")
            ax[1].plot(forecast, color=methods_colors[model_index], linestyle="--")

            ax[model_index + 2].plot(forecast, color=methods_colors[model_index], linestyle="--")
            axis_legend_copy = copy.copy(axis_legend)
            axis_legend_copy.append(nn_models_current_series[model_index].__name__ + " " + str(str("{:.2f}".format(smape))))
            ax[model_index + 2].legend(axis_legend_copy, prop={'size': legend_size})

            record_smapes[i-1].append(smape)

            if smape < best_smape:
                best_smape = smape
                best_model = nn_models_current_series[model_index]
                best_model_param = nn_models_params_current_series[model_index]

        print("--- SERIES "+str(i)+" SMAPES ---")

        for model_index in range(nbr_models_current_series):
            print("   MODEL "+nn_models_current_series[model_index].__name__+ " PARAM "+str(nn_models_params_current_series[model_index])+ " SMAPE "+str(record_smapes[i-1][model_index]))

        plt.tight_layout()
        plt.savefig(fig_folder+'series-' + str(i) + "-comparison.pdf")
        plt.show()


        """ --------------- now to chosen the best model -----------------------"""

        _, ax = plt.subplots(nrows=nbr_models_current_series + 2, ncols=1, figsize=(12, 10))

        ax[0].set_title('Comparison of model forecast on test series for series ' + str(i) + ' (last '+str(number_of_days)+' days)')
        ax[0].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
        ax[0].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        ax[1].set_title(
            'Comparison of model forecast on test series for series ' + str(i) + ' (whole series without outliers)')
        ax[1].plot(remove_outliers(data_train['series-' + str(i)]), color="black", linestyle="-")
        ax[1].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        axis_legend_all = ["Train series", "Validation series"]
        legend_size = 6
        axis_legend = ["Train series", "Validation series"]

        for model_index in range(nbr_models_current_series):
            ax[model_index + 2].set_title(
                'Comparison of model forecast on test series for series ' + str(i) + ' (last '+str(number_of_days)+' days)')
            ax[model_index + 2].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
            ax[model_index + 2].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

            smape, forecast = nn_models_current_series[model_index](data_train_full['series-' + str(i)],
                                                                    data_test,
                                                                    nn_models_params_current_series[model_index],
                                                                    horizon,
                                                                    del_outliers=True,
                                                                    normalize=True,
                                                                    plot=False)

            ax[0].plot(forecast, color=methods_colors[model_index], linestyle="--")
            ax[1].plot(forecast, color=methods_colors[model_index], linestyle="--")
            ax[model_index + 2].plot(forecast, color=methods_colors[model_index], linestyle="--")

            data_save_method = pd.DataFrame(index=interval_test)
            data_save_method['series-' + str(i)] = forecast
            data_save_method.to_csv(fig_folder+str(i)+"_"+nn_models_current_series[model_index].__name__+"_"+str(nn_models_params_current_series[model_index])+".csv")
            data_save_method = keyvalue(data_save_method)
            data_save_method.to_csv(fig_folder+str(i)+"_formatted_"+nn_models_current_series[model_index].__name__+"_"+str(nn_models_params_current_series[model_index])+".csv")

            if nn_models_current_series[model_index] == best_model:
                axis_legend_all.append(nn_models_current_series[model_index].__name__+ " (chosen)")
                axis_legend_copy = copy.copy(axis_legend)
                axis_legend_copy.append(nn_models_current_series[model_index].__name__ + " (chosen)")
                ax[model_index + 2].legend(axis_legend_copy, prop={'size': legend_size})

                data_submit['series-' + str(i)] = forecast

                chosen_method.append(best_model.__name__)
                chosen_method_smape.append(str("{:.2f}".format(best_smape)))
                chosen_method_param.append(best_model_param)
            else:
                axis_legend_all.append(nn_models_current_series[model_index].__name__)
                axis_legend_copy = copy.copy(axis_legend)
                axis_legend_copy.append(nn_models_current_series[model_index].__name__)
                ax[model_index + 2].legend(axis_legend_copy, prop={'size': legend_size})

        ax[0].legend(axis_legend_all, prop={'size': legend_size})
        ax[1].legend(axis_legend_all, prop={'size': legend_size})
        plt.tight_layout()
        plt.savefig(fig_folder+'series-' + str(i) + "-submission.pdf")
        plt.show()

    print("------------ SUBMISSION ------------")
    print()
    print(data_submit.to_string())
    print()
    print("------------ FORMATED SUBMISSION ------------")
    submission = keyvalue(data_submit)
    print(submission.to_string())
    submission.to_csv(save_path)
    print("------------ CHOSEN METHODS INFO FOR SUBMISSION ------------")
    print("METHODS "+str(chosen_method))
    print("PARAMS "+str(chosen_method_param))
    print("SMAPES "+str(chosen_method_smape))
    print()
    print("------------ ALL PREDICTED FORECAST ON ACTUAL DATASETS SAVED TO 'seriesid-nameofmethod-params.csv' ------------")

# compute,and_plot_submission("data/all_best_nn.csv", "save_info/")

def compute_and_plot_submission_with_arima(save_path, fig_folder):
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
    data_best_arima = pd.DataFrame(index=interval_test)

    # for plotting
    methods_colors = ["blue", "red", "cyan", "orange", "pink", "magenta", "grey", "yellow"]

    # to store the chosen model
    chosen_method = []
    chosen_method_smape = []
    chosen_method_param = []

    # to record the smape of each model
    record_smapes = []

    number_of_days = 62

    ARIMA_ORDER_VALID = []
    ARIMA_ORDER_TEST = []

    for i in range(1, nbr_series + 1):

        # to perform comparisons on models
        best_smape = 1000000000
        best_model = None
        best_model_param = None

        nn_models_current_series = models_to_try[i-1]
        nn_models_params_current_series = models_to_try_parameters[i-1]
        nbr_models_current_series = len(nn_models_current_series)
        record_smapes.append([])

        # for validation, we plot each validation forecast separately
        # axis 0 is a comparison of each forecast on last 90 days
        # axis 1 is a comparison of each forecast on the whole series
        _, ax = plt.subplots(nrows=nbr_models_current_series + 3, ncols= 1, figsize=(12, 10))

        ax[0].set_title('Comparison of model forecast on validation series for series '+str(i)+' (last '+str(number_of_days)+' days)')
        ax[0].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
        ax[0].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        ax[1].set_title('Comparison of model forecast on validation series for series '+str(i)+' (whole series without outliers)')
        ax[1].plot(remove_outliers(data_train['series-' + str(i)]), color="black", linestyle="-")
        ax[1].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        axis_legend = ["Train series", "Validation series", auto_arima_forecast.__name__]
        legend_size = 6
        for model_name in nn_models_current_series:
            axis_legend.append(model_name.__name__)
        ax[0].legend(axis_legend, prop={'size': legend_size})
        ax[1].legend(axis_legend, prop={'size': legend_size})
        axis_legend = ["Train series", "Validation series"]

        ax[2].set_title('Comparison of model forecast on validation series for series ' + str(i) + ' (last ' + str(
                number_of_days) + ' days)')
        ax[2].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
        ax[2].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        if i >= 100:
            smape, forecast, order, seasonal_order = auto_arima_forecast(data_train['series-' + str(i)],
                                                                    data_valid['series-' + str(i)],
                                                                    horizon,
                                                                    del_outliers=True,
                                                                    normalize=True,
                                                                    plot=False)
        else:
            order = ARIMA_parameters[i-1][0]
            seasonal_order = ARIMA_parameters[i-1][1]
            smape, forecast = arima_forecast(data_train['series-' + str(i)],
                                                                    data_valid['series-' + str(i)],
                                                                    horizon,
                                                                    order,
                                                                    seasonal_order,
                                                                    del_outliers=True,
                                                                    normalize=True,
                                                                    plot=False)
        ARIMA_ORDER_VALID.append((order,seasonal_order))
        ax[0].plot(forecast, color="steelblue", linestyle="--")
        ax[1].plot(forecast, color="steelblue", linestyle="--")

        ax[2].plot(forecast, color="steelblue", linestyle="--")
        axis_legend_copy = copy.copy(axis_legend)
        axis_legend_copy.append(auto_arima_forecast.__name__ + " "+str(order)+" "+str(seasonal_order)+" " + str(str("{:.2f}".format(smape))))
        ax[2].legend(axis_legend_copy, prop={'size': legend_size})

        record_smapes[i - 1].append(smape)

        if smape < best_smape:
            best_smape = smape
            best_model = auto_arima_forecast
            best_model_param = (order, seasonal_order)

        for model_index in range(nbr_models_current_series):
            ax[model_index + 3].set_title(
                'Comparison of model forecast on validation series for series ' + str(i) +' (last '+str(number_of_days)+' days)')
            ax[model_index + 3].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
            ax[model_index + 3].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

            smape, forecast = nn_models_current_series[model_index](data_train['series-' + str(i)],
                                                                    data_valid['series-' + str(i)],
                                                                    nn_models_params_current_series[model_index],
                                                                    horizon,
                                                                    del_outliers=True,
                                                                    normalize=True,
                                                                    plot=False)

            ax[0].plot(forecast, color=methods_colors[model_index], linestyle="--")
            ax[1].plot(forecast, color=methods_colors[model_index], linestyle="--")

            ax[model_index + 3].plot(forecast, color=methods_colors[model_index], linestyle="--")
            axis_legend_copy = copy.copy(axis_legend)
            axis_legend_copy.append(nn_models_current_series[model_index].__name__ + " " + str(str("{:.2f}".format(smape))))
            ax[model_index + 3].legend(axis_legend_copy, prop={'size': legend_size})

            record_smapes[i-1].append(smape)

            if smape < best_smape:
                best_smape = smape
                best_model = nn_models_current_series[model_index]
                best_model_param = nn_models_params_current_series[model_index]

        print("--- SERIES "+str(i)+" SMAPES ---")

        print("   MODEL " + auto_arima_forecast.__name__ + " PARAM " + str(ARIMA_ORDER_VALID[i-1]) + " SMAPE " + str(record_smapes[i - 1][0]))
        for model_index in range(nbr_models_current_series):
            print("   MODEL "+nn_models_current_series[model_index].__name__+ " PARAM "+str(nn_models_params_current_series[model_index])+ " SMAPE "+str(record_smapes[i-1][model_index+1]))

        plt.tight_layout()
        plt.savefig(fig_folder+'series-' + str(i) + "-comparison.pdf")
        plt.show()


        """ --------------- now to chosen the best model -----------------------"""

        _, ax = plt.subplots(nrows=nbr_models_current_series + 3, ncols=1, figsize=(12, 10))

        ax[0].set_title('Comparison of model forecast on test series for series ' + str(i) + ' (last '+str(number_of_days)+' days)')
        ax[0].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
        ax[0].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        ax[1].set_title(
            'Comparison of model forecast on test series for series ' + str(i) + ' (whole series without outliers)')
        ax[1].plot(remove_outliers(data_train['series-' + str(i)]), color="black", linestyle="-")
        ax[1].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

        axis_legend_all = ["Train series", "Validation series"]
        legend_size = 6
        axis_legend = ["Train series", "Validation series"]

        ax[2].set_title('Comparison of model forecast on test series for series ' + str(i) + ' (last ' + str(
                number_of_days) + ' days)')
        ax[2].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
        ax[2].plot(data_valid['series-' + str(i)], color="green", linestyle="-")


        if i >= 100:
            smape, forecast, order, seasonal_order = auto_arima_forecast(data_train_full['series-' + str(i)],
                                                                data_test,
                                                                horizon,
                                                                del_outliers=True,
                                                                normalize=True,
                                                                plot=False)

        else:
            order = ARIMA_parameters[i-1][0]
            seasonal_order = ARIMA_parameters[i-1][1]
            smape, forecast = arima_forecast(data_train_full['series-' + str(i)],
                                                                    data_test  ,
                                                                    horizon,
                                                                    order,
                                                                    seasonal_order,
                                                                    del_outliers=True,
                                                                    normalize=True,
                                                                    plot=False)

        ARIMA_ORDER_TEST.append((order,seasonal_order))

        data_best_arima['series-' + str(i)] = forecast

        ax[0].plot(forecast, color="steelblue", linestyle="--")
        ax[1].plot(forecast, color="steelblue", linestyle="--")

        ax[2].plot(forecast, color="steelblue", linestyle="--")

        data_save_method = pd.DataFrame(index=interval_test)
        data_save_method['series-' + str(i)] = forecast
        data_save_method.to_csv(fig_folder + str(i) + "_" + auto_arima_forecast.__name__ + "_" + str(order)+"_"+str(seasonal_order) + ".csv")
        data_save_method = keyvalue(data_save_method)
        data_save_method.to_csv(
            fig_folder + str(i) + "_formatted_" + auto_arima_forecast.__name__ + "_" + str(order)+"_"+str(seasonal_order) + ".csv")

        if auto_arima_forecast == best_model:
            axis_legend_all.append(auto_arima_forecast.__name__ + str(order)+" "+str(seasonal_order) +" (chosen)")
            axis_legend_copy = copy.copy(axis_legend)
            axis_legend_copy.append(auto_arima_forecast.__name__+ str(order)+" "+str(seasonal_order) +" (chosen)")
            ax[2].legend(axis_legend_copy, prop={'size': legend_size})

            data_submit['series-' + str(i)] = forecast

            chosen_method.append(best_model.__name__)
            chosen_method_smape.append(str("{:.2f}".format(best_smape)))
            chosen_method_param.append(str(order)+" - "+str(seasonal_order))
        else:
            axis_legend_all.append(auto_arima_forecast.__name__)
            axis_legend_copy = copy.copy(axis_legend)
            axis_legend_copy.append(auto_arima_forecast.__name__)
            ax[2].legend(axis_legend_copy, prop={'size': legend_size})


        for model_index in range(nbr_models_current_series):
            ax[model_index + 3].set_title(
                'Comparison of model forecast on test series for series ' + str(i) + ' (last '+str(number_of_days)+' days)')
            ax[model_index + 3].plot(data_train['series-' + str(i)][-number_of_days:], color="black", linestyle="-")
            ax[model_index + 3].plot(data_valid['series-' + str(i)], color="green", linestyle="-")

            smape, forecast = nn_models_current_series[model_index](data_train_full['series-' + str(i)],
                                                                    data_test,
                                                                    nn_models_params_current_series[model_index],
                                                                    horizon,
                                                                    del_outliers=True,
                                                                    normalize=True,
                                                                    plot=False)

            ax[0].plot(forecast, color=methods_colors[model_index], linestyle="--")
            ax[1].plot(forecast, color=methods_colors[model_index], linestyle="--")
            ax[model_index + 3].plot(forecast, color=methods_colors[model_index], linestyle="--")

            data_save_method = pd.DataFrame(index=interval_test)
            data_save_method['series-' + str(i)] = forecast
            data_save_method.to_csv(fig_folder+str(i)+"_"+nn_models_current_series[model_index].__name__+"_"+str(nn_models_params_current_series[model_index])+".csv")
            data_save_method = keyvalue(data_save_method)
            data_save_method.to_csv(fig_folder+str(i)+"_formatted_"+nn_models_current_series[model_index].__name__+"_"+str(nn_models_params_current_series[model_index])+".csv")

            if nn_models_current_series[model_index] == best_model:
                axis_legend_all.append(nn_models_current_series[model_index].__name__+ " (chosen)")
                axis_legend_copy = copy.copy(axis_legend)
                axis_legend_copy.append(nn_models_current_series[model_index].__name__ + " (chosen)")
                ax[model_index + 3].legend(axis_legend_copy, prop={'size': legend_size})

                data_submit['series-' + str(i)] = forecast
                chosen_method.append(best_model.__name__)
                chosen_method_smape.append(str("{:.2f}".format(best_smape)))
                chosen_method_param.append(best_model_param)
            else:
                axis_legend_all.append(nn_models_current_series[model_index].__name__)
                axis_legend_copy = copy.copy(axis_legend)
                axis_legend_copy.append(nn_models_current_series[model_index].__name__)
                ax[model_index + 3].legend(axis_legend_copy, prop={'size': legend_size})

        ax[0].legend(axis_legend_all, prop={'size': legend_size})
        ax[1].legend(axis_legend_all, prop={'size': legend_size})

        plt.tight_layout()
        plt.savefig(fig_folder+'series-' + str(i) + "-submission.pdf")
        plt.show()

    print("------------ SUBMISSION ------------")
    print()
    print(data_submit.to_string())
    data_submit.to_csv("data/all_best_nn_nosub.csv")
    print()
    print("------------ FORMATED SUBMISSION ------------")
    submission = keyvalue(data_submit)
    print(submission.to_string())
    submission.to_csv(save_path)
    print("------------ CHOSEN METHODS INFO FOR SUBMISSION ------------")
    print("METHODS "+str(chosen_method))
    print("PARAMS "+str(chosen_method_param))
    print("SMAPES "+str(chosen_method_smape))
    print()
    print("------------ ARIMA ORDERS ------------")
    print("ARIMA ORDER VALID "+str(ARIMA_ORDER_VALID))
    print("ARIMA ORDER TEST "+str(ARIMA_ORDER_TEST))
    print()
    print("------------ ARIMA SUBMISSION AN FORMATED SUBMISSION ------------")
    print()
    print(data_best_arima.to_string())
    data_best_arima.to_csv("data/arima_best.csv")
    print()
    submission = keyvalue(data_best_arima)
    print(submission.to_string())
    submission.to_csv("data/arima_best_submission.csv")
    print()
    print("------------ ALL PREDICTED FORECAST ON ACTUAL DATASETS SAVED TO 'seriesid-nameofmethod-params.csv' ------------")

compute_and_plot_submission_with_arima("data/all_best_nn.csv", "save_info/")