import keras.backend as K
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def embedding(data, p):
    data_shifted = data.copy()
    for lag in range(-p + 1, 2):
        data_shifted['y_t' + '{0:+}'.format(lag)] = data_shifted['y'].shift(-lag, freq='D')
    data_shifted = data_shifted.dropna(how='any')
    y = data_shifted['y_t+1'].to_numpy()
    X = data_shifted[['y_t' + '{0:+}'.format(lag) for lag in range(-p + 1, 1)]].to_numpy()
    return X, y, data_shifted


def smape(y_true, y_pred):
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


def smapeLSTM(true,predicted):
    return K.abs(predicted - true) / K.maximum(K.abs(true) + K.abs(predicted) + 0.1, 0.5 + 0.1) * 2.0


def keyvalue(df):
    df["horizon"] = range(1, df.shape[0] + 1)
    res = pd.melt(df, id_vars=["horizon"])
    res = res.rename(columns={"variable": "series"})
    res["Id"] = res.apply(lambda row: "s" + str(row["series"].split("-")[1]) + "h" + str(row["horizon"]), axis=1)
    res = res.drop(['series', 'horizon'], axis=1)
    res = res[["Id", "value"]]
    res = res.rename(columns={"value": "forecasts"})
    return res


def main():
    data = pd.read_csv("/home/damien/HandsOnAI2019/Challenges/Defi3/train.csv", index_col="Day")

    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
    data = data.asfreq('d')

    interval_train = pd.date_range(start='2015-07-01', end='2017-06-30')
    interval_valid = pd.date_range(start='2017-07-1', end='2017-08-20')

    interval_test = pd.date_range(start='2017-08-21', end='2017-09-10')
    HORIZON = len(interval_test)

    data_train = data.loc[interval_train]
    data_valid = data.loc[interval_valid]

    forecasts_nn = pd.DataFrame(index=interval_test)

    for iseries in data_train.columns:
        print(iseries)

        train = data_train[iseries].to_frame()
        valid = data_valid[iseries].to_frame()

        BATCH_SIZE = 32  # number of samples per mini-batch
        EPOCHS = 2500  # maximum number of times the training algorithm will cycle through all samples

        p = 15

        scaler = MinMaxScaler()

        train["y"] = scaler.fit_transform(train)
        X_train, y_train, train_embedded = embedding(train, p)

        valid['y'] = scaler.transform(valid)
        X_valid, y_valid, valid_embedded = embedding(valid, p)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="relu"))
        model.compile(loss=smapeLSTM, optimizer="adam")

        best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), callbacks=[earlystop], verbose=2)

        x = X_valid[-1, 1:]
        ypred = y_valid[-1]

        f_nn = []
        for horizon in range(1, HORIZON + 1):
            x_test = np.expand_dims(np.append(x, ypred), axis=0)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            ypred = model.predict(x_test)
            f_nn.append(float(ypred))

        # f_nn = []
        # tests = []
        # for horizon in range(1, HORIZON + 1):
        #     if len(tests) > 7:
        #         x_test = np.expand_dims(np.append(x, tests[-7]), axis=0)
        #     else:
        #         x_test = np.expand_dims(np.append(x, y_valid[horizon-7]), axis=0)
        #     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        #     ypred = model.predict(x_test)
        #     f_nn.append(float(ypred))
        #     tests.append(ypred)

        f_nn = np.expand_dims(f_nn, axis=0)
        forecasts_nn[iseries] = scaler.inverse_transform(f_nn).flatten()

    pred_nn = keyvalue(forecasts_nn)
    pred_nn.to_csv("submissionLSTMRelu2500.csv", index=False)


if __name__ == "__main__":
    main()
