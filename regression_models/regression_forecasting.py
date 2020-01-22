# regression algorithms lives here, the file is thought to be used with
# any python regression algorithm that uses '.fit' and '.predict' methods

# import necessary modules
import xgboost as xgb

import matplotlib.pyplot as plt

import sklearn.kernel_ridge
from sklearn import preprocessing
from sklearn import base

from regression_tools import *
from regression_preprocessing import *

####################################################################################################
####################################################################################################


# split an array in a series of examples (X,y) with X having length f_steps_in
# and y length steps_out
def big_array_splitter(f_ar, f_steps_in, f_steps_out):

    f_in_out_length = f_steps_in + f_steps_out
    limit = len(f_ar[0]) - f_in_out_length

    # still big array with 45 columns
    res_X = []
    res_y = []
    for f_s in range(45):  # TODO: this is a magic number, to change
        Xs = []
        Ys = []
        for d in range(limit):
            in_left = d
            in_right = d + f_steps_in
            out_left = d + f_steps_in
            out_right = d + f_steps_in + f_steps_out
            xs = f_ar[f_s][in_left: in_right].copy()
            ys = f_ar[f_s][out_left: out_right].copy()

            Xs.append(xs)
            Ys.append(ys)
        res_X.append(Xs)
        res_y.append(Ys)

    res_X = np.array(res_X)
    res_y = np.array(res_y)

    return res_X, res_y


# explained below
def demerge_dataset(f_dataset, f_steps_out):
    """
    demerge_dataset transform a dataset of examples in several as follows:
    let (x_1, x_2) -> (y_1, y_2) be an example and
    steps_out = 2
    then demerge_dataset will make two new examples:
    -(x_1, x_2) -> (y_1)
    -(x_1, x_2) -> (y_2)
    """

    [f_X_train, f_X_val, f_y_train, f_y_val] = f_dataset
    res_demerged_dataset = []

    for f_i in range(f_steps_out):
        f_train, f_val = f_y_train[:, f_i].copy(), f_y_val[:, f_i].copy()

        res_demerged_dataset.append([f_X_train, f_X_val, f_train, f_val])

    res_demerged_dataset = np.array(res_demerged_dataset)
    return res_demerged_dataset


def train_for_one_step(f_dataset, f_model, f_steps_out, f_params=None, f_clone=False):
    """
    train_for_one_step follows demerge_dataset, it creates and train a number
    of steps_out models, one for each coordinate in the target vectors
    """

    if f_params is None:
        f_params = {}

    res_demerged_dataset = demerge_dataset(f_dataset, f_steps_out)

    res_models = []

    for f_i in range(f_steps_out):

        if f_clone:
            res_m = base.clone(f_model)
        else:
            res_m = f_model(**f_params)

        [f_X_train, _, f_t_train, _] = res_demerged_dataset[f_i]

        res_m.fit(f_X_train, f_t_train)

        res_models.append(res_m)

    return res_models


def train_for_one_step_all_series(f_dataset, f_model, f_steps_out, f_params=None, f_clone=False):
    """
    train_for_one_step_all_series is a wrapper for the train_for_one_step
    function
    :param f_dataset:
    :param f_model:
    :param f_steps_out:
    :param f_params:
    :return:
    """

    res_models_by_series = []

    for f_i in range(len(f_dataset)):
        res_models = train_for_one_step(f_dataset[f_i], f_model, f_steps_out, f_params, f_clone)
        res_models_by_series.append(res_models)

    return res_models_by_series


def predict_custom_out(f_data, f_models, f_steps_out):
    """
    take a list of models with the same lenght as steps_out and predict a
    vector of length steps_out
    :param f_data:
    :param f_models:
    :param f_steps_out:
    :return:
    """
    res_data = f_data.copy()
    res_preds = []

    for f_i in range(f_steps_out):
        f_m = f_models[f_i]

        t_preds = f_m.predict(res_data)

        res_preds.append(t_preds)

    res_preds = np.array(res_preds)
    res_preds = np.transpose(res_preds)

    return res_preds


def predict_custom_out_all_series(f_data, f_models_by_series, f_steps_out):
    """
    wrapper for the predict_custom_out function
    :param f_data:
    :param f_models_by_series:
    :param f_steps_out:
    :return:
    """
    res_preds = []

    for f_i in range(len(f_data)):
        res_p = predict_custom_out(f_data[f_i], f_models_by_series[f_i], f_steps_out)

        res_preds.append(res_p)

    res_preds = np.array(res_preds)
    return res_preds


# this function creates examples that take into account not only the
# immediate past, but even further periods
def to_regression_examples(f_array, f_steps_in, f_steps_out,
                           f_series,
                           f_days,
                           f_test_size,
                           f_splitter_method=split_to_test,
                           f_periods=None):
    if f_periods is None:
        f_periods = []

    # create list with all periods and include the zero period
    res_periods = [0]
    res_periods.extend(f_periods)
    res_periods.sort()
    greater_period = max(res_periods)

    (res_X, res_y) = big_array_splitter(f_array, f_steps_in, f_steps_out)

    # this part of the code modifies the examples to take into account the
    # past, we see one year in the past and six months in the past
    res_length_X = len(res_X[0])

    # create new array for the examples
    f_periods_length = len(f_periods)
    res_new_X = np.zeros((series, res_length_X - greater_period, steps_in * (1 + f_periods_length)))

    # populate the array with the concatenation of values
    for f_s in range(f_series):
        for f_i in range(len(res_new_X[0])):

            list_of_periods_to_concatenate = [res_X[f_s][f_i + f_p] for f_p in res_periods]
            res_new_X[f_s][f_i] = np.concatenate(list_of_periods_to_concatenate)

    # create a new array for the target examples with the values that
    # correspond to the new_X array

    res_new_y = np.zeros((f_series, res_length_X - greater_period, f_steps_out))

    for f_s in range(f_series):
        for f_i in range(len(res_new_X[0])):
            res_new_y[f_s][f_i] = res_y[f_s][f_i + greater_period]

    # create the dataset, one per series
    res_dataset = f_splitter_method(res_new_X, res_new_y, f_days, f_series, f_steps_in,
                                    f_steps_out, f_test_size)

    return res_dataset


def create_validation_dataframe(f_df, f_steps_in, f_series, f_periods=None, f_offset=0):
    if f_periods is None:
        f_periods = []
    # immediate last steps_in days

    list_of_validation_dataframes = []
    res_periods = [0]
    res_periods.extend(f_periods)
    res_periods.sort(reverse=True)
    res_periods_length = len(res_periods)

    for res_p in res_periods:
        left_bound = - (f_steps_in + f_offset + res_p)
        right_bound = - (f_offset + res_p)

        # depends of the offset
        if right_bound == 0:
            validation_dataframe = f_df.iloc[left_bound:].copy()
        else:
            validation_dataframe = f_df.iloc[left_bound: right_bound].copy()

        validation_dataframe = to_array(validation_dataframe)
        list_of_validation_dataframes.append(validation_dataframe)

    res_X_validation = np.zeros((f_series, f_steps_in * res_periods_length))
    for f_s in range(f_series):
        list_to_concatenate = [validation_dataframe[f_s] for validation_dataframe in
                               list_of_validation_dataframes]
        res_X_validation[f_s] = np.concatenate(list_to_concatenate)

    res_y_validation_dataframe = f_df.tail(f_offset).copy()

    return res_X_validation, res_y_validation_dataframe

####################################################################################################
####################################################################################################


if __name__ == '__main__':

    ############################################################################
    ############################################################################
    # save data to file or plot
    plot_to_file = True
    show_plot = False

    # custom values for this run of training
    steps_in = 14  # length of the entry vector for the models
    steps_out = 21  # length of the target vector
    test_size = 0.1

    # periods, for using the past in the regression methods
    p_periods = [182, 365]

    path_to_data = 'train.csv'

    scaler_to_use = preprocessing.MinMaxScaler()

    # these are some models that we use for the regression
    # a simple one, a medium one and a complex one
    # you can of course use a single model, but 'models_constructors' has to be
    # list
    several_models = True
    models_constructors = [sklearn.linear_model.LinearRegression,
                           sklearn.linear_model.Ridge,
                           sklearn.kernel_ridge.KernelRidge,
                           xgb.XGBRegressor]

    # prefix to be attached to submissions
    submission_prefix = "my_sub08"

    # colors for plotting
    colors = ['red', 'green', 'cyan', 'magenta', 'yellow', 'brown']
    ############################################################################
    ############################################################################

    print("""
>>> Regression modeling of dataset: {}.
>>> Parameters:
    - plot_to_file: {},
    - show_plot: {},
    - steps_in: {},
    - steps_out: {},
    - test_size: {},
    - past periods: {},
    - scale method: {},
    - regression methods: {}.
    """.format(path_to_data,
               plot_to_file,
               show_plot,
               steps_in,
               steps_out,
               test_size,
               p_periods,
               scaler_to_use,
               models_constructors))

    # prepare the data

    # read the csv data
    df = pd.read_csv(path_to_data, parse_dates=['Day'], index_col='Day')

    # remove outliers
    dfc = df.copy(deep=True)
    dfc = remove_outliers(dfc)

    # scale and keep track of the scaler
    (dfc, scaler) = scale(dfc, scaler_to_use)

    # keep track of the original shape of the dataframe
    print(">>> shape of data: {}".format(tuple(dfc.shape)))
    (days, series) = tuple(dfc.shape)

    # transform the data to a numpy array
    data = to_array(dfc)

    # create the datasets, one per serie
    datasets = to_regression_examples(data, steps_in, steps_out, series, days,
                                      test_size,
                                      f_periods=p_periods,
                                      f_splitter_method=split_to_test2)

    print(">>> datasets created with shape: {}".format(tuple(datasets.shape)))
    ############################################################################
    ############################################################################

    # fit models to the data
    print(">>> modeling fit")
    models_by_series_list = []
    # train each model

    if several_models:
        print(">>> fitting several models...")
        for model in models_constructors:
            models_by_series = train_for_one_step_all_series(datasets, model,
                                                             steps_out)
            models_by_series_list.append(models_by_series)
    else:
        print('>>> only one model selected, fitting...')

        # need to define special model here
        submission_prefix = 'mysub09_'
        submission_prefix += 'linear_regression_alone'

        model = sklearn.kernel_ridge.KernelRidge()

        models_constructors = [model]

        print(">>> model: {}".format(str(model)))

        models_by_series = train_for_one_step_all_series(datasets, model,
                                                         steps_out,
                                                         f_clone=True)
        models_by_series_list.append(models_by_series)

    ############################################################################
    ############################################################################
    # create a validation dataframe
    X_validation, y_validation_dataframe = create_validation_dataframe(dfc, steps_in, series,
                                                                       f_periods=p_periods,
                                                                       f_offset=steps_out)

    X_validation = np.array([x.reshape((1, -1)) for x in X_validation])

    # predict the validation dataframe
    validation_futures = []

    print(">>> predict validation set")
    # one by model in models_constructors
    for models_by_series in models_by_series_list:
        X_validation_preds = predict_custom_out_all_series(X_validation,
                                                           models_by_series, steps_out)
        X_validation_preds = X_validation_preds.reshape((series, steps_out))
        X_validation_preds = np.transpose(X_validation_preds)

        # create the dataframe
        validation_future = pd.DataFrame(X_validation_preds,
                                         index=y_validation_dataframe.index)

        # rename columns
        validation_future = validation_future.rename(columns=create_rename_dic())

        # unscale
        validation_future_unscaled = unscale(validation_future, scaler)
        validation_futures.append(validation_future_unscaled)

    print(">>> compute the smape score by series and models")
    # compute the smape score and put it in an array, one per series
    number_of_models = len(models_by_series_list)
    scores = np.zeros((number_of_models, series))
    for i in range(number_of_models):
        scores[i] = compute_score(df.copy(), validation_futures[i], steps_out, smape_loss)

    weighted_validation = weighted_prediction(scores, validation_futures)
    weighted_score = compute_score(df.copy(), weighted_validation, steps_out, smape_loss)


    print(">>> plot and safe the scores")
    plt.figure(figsize=(16, 10))
    for i in range(len(models_constructors)):
        plt.plot(scores[i], color=colors[i])

    plt.plot(weighted_score, color=colors[len(models_constructors) + 1])
    legend_for_scores = [str(m) for m in models_constructors]
    legend_for_scores.append("weighted prediction")
    plt.legend(legend_for_scores)
    plt.title("Smape scores")
    plt.xlabel("series")
    plt.ylabel("smape score")
    if plot_to_file:
        plt.savefig(('plot_pdf/' + submission_prefix + '_smape_scores.pdf'))
    if show_plot:
        plt.show()

    print(">>> predict the future")
    # predict the future
    X_future, _ = create_validation_dataframe(dfc, steps_in, series, f_periods=p_periods,
                                              f_offset=0)

    X_future = np.array([x.reshape((1, -1)) for x in X_future])

    preds_futures = []
    for models_by_series in models_by_series_list:
        preds_future = predict_custom_out_all_series(X_future, models_by_series,
                                                     steps_out)

        preds_future = preds_future.reshape((series, steps_out))
        preds_future = np.transpose(preds_future)

        preds_future = pd.DataFrame(preds_future,
                                    index=create_next_index(dfc, steps_out))
        preds_future = preds_future.rename(columns=create_rename_dic())
        preds_future = unscale(preds_future, scaler)

        preds_futures.append(preds_future)

    print(">>> new prediction using a weighted mean of the predictions")
    # create a new prediction with a weighted mean of all others predictions
    weighted_future = weighted_prediction(scores, preds_futures)

    print(">>> plot and save to file if necessary")
    # plot validation and predictions

    preds_futures_length = len(preds_futures)

    for column in validation_futures[0].columns:
        plt.figure(figsize=(16, 10))
        plt.plot(df.tail(100)[column], color='blue')
        plt.plot(weighted_future[column], color='black')

        for i in range(preds_futures_length):
            plt.plot(validation_futures[i][column],
                     color=colors[i % preds_futures_length])
            plt.plot(preds_futures[i][column],
                     color=colors[i % preds_futures_length])

        legend = ['real', 'weighted']
        legend.extend([str(m) for m in models_constructors])
        plt.legend(legend)

        plt.title('Predictions of ' + str(column))

        if plot_to_file:
            plt.savefig(('plot_pdf/' + submission_prefix +
                         'validation_plus_predictions_of_'
                         + str(column) + '.pdf'))

        if show_plot:
            plt.show()

    ############################################################################
    ############################################################################

    # now save submissions to csv files
    path_to_submissions = 'submissions'
    for i in range(len(models_constructors)):
        method_name = str(models_constructors[i])
        future = preds_futures[i]
        name_of_submission = (path_to_submissions + "/" + submission_prefix
                              + "_" + method_name + '.csv')
        sub = keyvalue(future)
        sub.to_csv(name_of_submission, index=False)

    # TODO: do this tomorrow
    # also weighted submission
    sub = keyvalue(weighted_future)
    name_of_submission = path_to_submissions + "/" + submission_prefix + "_" + 'weighted' + '.csv'
    sub.to_csv(name_of_submission, index=False)
