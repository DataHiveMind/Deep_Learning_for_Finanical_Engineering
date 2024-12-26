import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import keras
from keras.models import Sequential
from keras.layers import Dense

def basic_keras_model(data, test_data, predictors, target):
    model = Sequential()
    n_cols = data.shape[1]
    model.add(Dense(5, activition = 'relu', input_shape = (n_cols, )))
    model.add(Dense(5, activition = 'relu'))
    model.add(Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(predictors, target)
    predictions = model.predict(test_data)
    return predictions

def build_lstm_model(input_shape, units):
    """Builds an LSTM model.

    Args:
        units: number of units for model
        input_shape: Shape of the input data.

    Returns:
        Compiled LSTM model.
    """

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=units))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ is "__main__":  
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32)