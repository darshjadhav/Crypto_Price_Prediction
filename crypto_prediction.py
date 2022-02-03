from cProfile import label
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

def read_data(cryptopair, future_days):
    
    # Training Data
    start = dt.datetime(2016,1,1)
    # end = dt.datetime(2021,1,1)
    end = dt.datetime.now()

    coin = pdr.DataReader(cryptopair, 'yahoo', start, end)
    dateindex = coin.index
    scalar = MinMaxScaler(feature_range=(0,1))
    coindata = scalar.fit_transform(coin["Close"].values.reshape(-1,1))

    day_range = 30
    
    x_train = []
    y_train = []

    for i in range(day_range, len(coindata)-future_days):
        x_train.append(coindata[i - day_range:i, 0]) # Get data between day range
        y_train.append(coindata[i + future_days, 0]) # Get close value at day 'n'
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Testing Data
    test_start = dt.datetime(2021,1,2)
    test_end = dt.datetime.now()

    testingdata = pdr.DataReader(cryptopair, 'yahoo', test_start, test_end)
    y_test = testingdata["Close"].values
    testdata = scalar.fit_transform(y_test.reshape(-1,1))

    x_test = []

    for i in range(day_range, len(testdata)):
        x_test.append(testdata[i-day_range:i,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, scalar, dateindex
    


def model(x_train, y_train, x_test, scalar):
    # Model Architecture
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Train Model
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=2, batch_size=32)

    # Test Model

    y_pred = model.predict(x_test)
    y_pred = scalar.inverse_transform(y_pred)

    return y_pred



def plot_results(cryptopair, future_days, dateindex, y_pred, y_test):
    future_bias = np.zeros((future_days,1))
    predictions = np.concatenate((future_bias,y_pred), axis=0)
    # print(y_test.shape)
    # print(predictions.shape)
    # print(dateindex.shape)
    datevals = dateindex[dateindex.shape[0] - y_test.shape[0]:dateindex.shape[0]]
    plt.figure()
    plt.plot(datevals, y_test, color="black", label="Actual Prices")
    plt.plot(datevals, predictions, color="red", label="Predicted Prices")
    plt.title(f"{cryptopair} Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.show()

    

def main():
    cryptopair = "ETH-USD"
    future_days = 30
    x_train, y_train, x_test, y_test, scalar, dateindex = read_data(cryptopair, future_days)
    predicted_vals = model(x_train, y_train, x_test, scalar)
    plot_results(cryptopair, future_days, dateindex, predicted_vals, y_test)



if __name__ == "__main__":
    main()