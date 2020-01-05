# !pip install pmdarima

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima import auto_arima

# For NEURAL NETWORK
#from keras import regularizers
#from keras.models import Model, Sequential
#from keras.layers import Dense
#from keras.callbacks import EarlyStopping, ModelCheckpoint

# For ML
from sklearn.preprocessing import MinMaxScaler


# from utils import keyvalue, embedding, smape
# import utils
# !pylint utils

def embedding(data, p):
    data_shifted = data.copy()
    for lag in range(-p + 1, 2):
        data_shifted['y_t' + '{0:+}'.format(lag)] = data_shifted['y'].shift(-lag, freq='D')
    data_shifted = data_shifted.dropna(how='any')
    y = data_shifted['y_t+1'].to_numpy()
    X = data_shifted[['y_t' + '{0:+}'.format(lag) for lag in range(-p + 1, 1)]].to_numpy()
    return X, y, data_shifted


def smape(y_true, y_pred):
    """
    Error function
    :param y_true:
    :param y_pred:
    :return:
    """
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


def keyvalue(df):
    df["horizon"] = range(1, df.shape[0] + 1)
    res = pd.melt(df, id_vars=["horizon"])
    res = res.rename(columns={"variable": "series"})
    res["Id"] = res.apply(lambda row: "s" + str(row["series"].split("-")[1]) + "h" + str(row["horizon"]), axis=1)
    res = res.drop(['series', 'horizon'], axis=1)
    res = res[["Id", "value"]]
    res = res.rename(columns={"value": "forecasts"})
    return res

'''
# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/hands-on-ai-umons-2019/train.csv", index_col="Day")

data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
data = data.asfreq('d')

interval_train = pd.date_range(start='2015-07-01', end='2017-06-30')
interval_valid = pd.date_range(start='2017-07-1', end='2017-08-20')

interval_test = pd.date_range(start='2017-08-21', end='2017-09-10')
HORIZON = len(interval_test)

data_train = data.loc[interval_train]
data_valid = data.loc[interval_valid]

forecasts_autoarima = pd.DataFrame(index=interval_test)
forecasts_naive = pd.DataFrame(index=interval_test)
forecasts_nn = pd.DataFrame(index=interval_test)
'''
"""
for iseries in data_train.columns:
    print(iseries)
    series_train = data_train[iseries]

    '''
    ##### auto.arima
    model = auto_arima(series_train, start_p=1, start_q=1,
                               max_p=3, max_q=3, m = 7,
                               start_P=0, seasonal=True,
                               d=1, D=1, trace=True,
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)
    model.fit(series_train)
    f_autoarima = model.predict(n_periods = HORIZON)
    #f_autoarima = pd.DataFrame(f_autoarima, index = interval_test, columns=['pred'])
    forecasts_autoarima[iseries] = f_autoarima
    '''

    # naive seasonal
    f_naive = [series_train[-(7 + h - 1)] for h in range(1, HORIZON + 1)]
    forecasts_naive[iseries] = f_naive

    '''
    # 
    train = data_train[iseries].to_frame()
    valid = data_valid[iseries].to_frame()

    #####  MLP (recursive)
    LATENT_DIM = 5 # number of units in the dense layer
    BATCH_SIZE = 32 # number of samples per mini-batch
    EPOCHS = 100 # maximum number of times the training algorithm will cycle through all samples
    
    p = 15
    OUTPUT_LENGTH = 1

    scaler = MinMaxScaler()
    
    train["y"] = scaler.fit_transform(train)
    X_train, y_train, train_embedded = embedding(train, p)

    valid['y'] = scaler.transform(valid)
    X_valid, y_valid, valid_embedded  = embedding(valid, p)

    model = Sequential()
    model.add(Dense(LATENT_DIM, activation="relu", input_shape=(p,)))
    model.add(Dense(OUTPUT_LENGTH))
    model.compile(optimizer='RMSprop', loss='mse')
        
    best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
    history = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_valid, y_valid),
                    callbacks=[earlystop, best_val],
                    verbose=1)
    
    #best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)
    #best_epoch = np.argmin(np.array(history.history['val_loss']))+1
    #model.load_weights("model_{:02d}.h5".format(best_epoch))
    #plot_df = pd.DataFrame.from_dict({'train_loss':history.history['loss'], 'val_loss':history.history['val_loss']})
    #plot_df.plot(logy=True, figsize=(10,10), fontsize=12)
    #plt.xlabel('epoch', fontsize=12)
    #plt.ylabel('loss', fontsize=12)
    #plt.show()
    
    # recursive forecasts
    x = X_valid[-1, 1:]
    ypred = y_valid[-1]
     
    f_nn = []
    for horizon in range(1, HORIZON+1):
        x_test = np.expand_dims(np.append(x, ypred), axis=0)
        ypred = model.predict(x_test)
        x = x_test[x_test.shape[0] - 1, 1:]
        f_nn.append(float(ypred))
    
    f_nn = np.expand_dims(f_nn, axis = 0)
    forecasts_nn[iseries] = scaler.inverse_transform(f_nn).flatten()
    '''
"""
'''
pred_naive = keyvalue(forecasts_naive)
pred_naive.to_csv("submission_naive.csv", index=False)
'''
data_path = "data/train.csv"

def setup_data(data_path):
    test = pd.read_csv("data/submission_arima.csv")
    test = keyvalue(test)
    test.to_csv("submission_arima.csv", index=False)
    assert True == False

    # with this dataset, the index column is day
    data = pd.read_csv(data_path, index_col="Day")

    # changing index to datetime object year - month - day
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")

    # note sure what for
    data = data.asfreq('d')

    nbr_series = len(data.columns)
    nbr_samples = data["series-1"].count()

    print(nbr_series)
    print(nbr_samples)

    start_date = data.index[0]
    end_date = data.index[781]

    print(start_date)
    print(end_date)

    interval_train = pd.date_range(start=start_date, end='2017-06-30')
    interval_valid = pd.date_range(start='2017-07-1', end=end_date)
    interval_test = pd.date_range(start='2017-08-21', end='2017-09-10')

    # number of samples we are predicting
    horizon = len(interval_test)

    # separating data into train and validation set
    data_train = data.loc[interval_train]
    data_valid = data.loc[interval_valid]

    forecasts_autoarima = pd.DataFrame(index=interval_test)
    """
    for serie in data_train.columns:

        print(serie)

        # select column for training
        series_train = data_train[serie]

        # naive seasonal
        #f_naive = [series_train[-(7 + h - 1)] for h in range(1, HORIZON + 1)]
        #forecasts_naive[iseries] = f_naive

        model = auto_arima(series_train, start_p=1, start_q=1,
                           max_p=2, max_q=2, m=7,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
        model.fit(series_train)

        f_autoarima = model.predict(n_periods=horizon)
        forecasts_autoarima[serie] = f_autoarima
    """
    for serie in data_train.columns:
        print(serie)

        # select column for training
        series_train = data_train[serie]

        # naive seasonal
        # f_naive = [series_train[-(7 + h - 1)] for h in range(1, HORIZON + 1)]
        # forecasts_naive[iseries] = f_naive


        med = np.median(series_train)
        q25, q75 = np.percentile(series_train, 25), np.percentile(series_train, 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        upper = q75 + cut_off
        outliers = series_train > upper
        series_train[outliers] = np.nan
        series_train.fillna(med, inplace=True)

        model = auto_arima(series_train, start_p=1, start_q=1,
                               max_p=2, max_q=2, m=7,
                               start_P=0, seasonal=True,
                               d=1, D=1, trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
        model.fit(series_train)

        f_autoarima = model.predict(n_periods=horizon)
        forecasts_autoarima[serie] = f_autoarima

    forecasts_autoarima.to_csv("submission_arima.csv", index=False)
setup_data(data_path)

assert True == False
