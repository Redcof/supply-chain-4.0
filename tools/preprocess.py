# Creating lag features for time-series data
import logging

import pandas as pd
from sklearn import preprocessing

logger = logging.getLogger(__name__)


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


def timeseries_split(data, target, train_size=.8):
    logger.info("Splitting dataset..")
    # Splitting time-series data into training and testing sets
    train_size = int(len(data) * train_size)
    data_train, data_test = data[:train_size], data[train_size:]
    y_train = data_train[target]
    y_test = data_test[target]
    x_train = data_train.drop(columns=[target])
    x_test = data_test.drop(columns=[target])
    return x_train, x_test, y_train, y_test


def perform_label_encoding(df):
    for col in df.columns:
        if df[col].dtype == 'category':
            label_encoder = preprocessing.LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            df[col] = df[col].astype('int')
    return df


def preprocess(data, target, datetime_col=None, format_str=None, is_extra_feature_enabled=True):
    """
    Creates target_lag, target_rolling_mean, target_fft, and datatime features
    :param is_extra_feature_enabled:
    :param data:
    :param target:
    :param datetime_col:
    :param format_str:
    :return:
    """
    if is_extra_feature_enabled:
        data = create_lag_features(data, target)
        data = create_rolling_mean(data, target)
        data = apply_fourier_transform(data, target)
    if datetime_col:
        data = process_datetime(data, datetime_col, format_str)
    data = perform_label_encoding(data)
    return data


def process_datetime(df, datetime_col, format_str=None):
    """
    Creates day, month, hour, week_day_name columns
    :param df:
    :param datetime_col:
    :param format_str:
    :return:
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col], format=format_str)
    df['day'] = df[datetime_col].dt.day.astype('category')
    df['month'] = df[datetime_col].dt.month.astype('category')
    df['hour'] = df[datetime_col].dt.hour.astype('category')
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['day_of_week'] = df['day_of_week'].map({
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }).astype('category')
    return df
