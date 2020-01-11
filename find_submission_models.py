import copy

from nn_multi_step_forecasting import nn_multi_step_forecast, nn_with_past_multi_step_forecast, \
    nn_with_past_outliers_multi_step_forecast
from nn_single_step_forecasting import nn_single_step_forecast, nn_with_past_outliers_single_step_forecast, \
    nn_with_past_single_step_forecast

from matplotlib import pyplot as plt

import pandas as pd


# These models and parameters have been chosen through experiments
from utils import remove_outliers, keyvalue

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

compute_and_plot_submission("data/all_best_nn.csv", "save_info/")