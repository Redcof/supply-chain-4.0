# Creating lag features for time-series data

def create_lag_features(data, column, lag_steps=3):
    for i in range(1, lag_steps + 1):
        data[f'lag_{i}'] = data[column].shift(i)

    return data


# Creating rolling mean for time-series data

def create_rolling_mean(data, column, window_size=3):
    data['rolling_mean'] = data[column].rolling(window=window_size).mean()

    return data


# Applying Fourier transformation for capturing seasonality

import numpy as np
from scipy.fft import fft


def apply_fourier_transform(data, column):
    values = data[column].values

    fourier_transform = fft(values)

    data['fourier_transform'] = np.abs(fourier_transform)

    return data


def split(data, target, train_size=.8):
    # Splitting time-series data into training and testing sets

    train_size = int(len(data) * train_size)
    data = create_lag_features(data, target)
    data = create_rolling_mean(data, target)
    data = apply_fourier_transform(data, target)
    data_train, data_test = data[:train_size], data[train_size:]
    y_train = data_train[target]
    y_test = data_test[target]
    X_train = data_train.drop(columns=[target])
    X_test = data_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test
