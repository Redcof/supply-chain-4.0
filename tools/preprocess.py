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
    logger.info("Performing label encoding...")
    for col in df.columns:
        if df[col].dtype in ('category', 'string'):
            label_encoder = preprocessing.LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            df[col] = df[col].astype('int')
    return df


def preprocess(data, target, datetime_col=None, format_str=None, is_extra_feature_enabled=True, label_encoding=True):
    """
    Creates target_lag, target_rolling_mean, target_fft, and datatime features
    :param is_extra_feature_enabled:
    :param data:
    :param target:
    :param datetime_col:
    :param format_str:
    :param label_encoding:
    :return:
    """
    if is_extra_feature_enabled:
        data = create_lag_features(data, target)
        data = create_rolling_mean(data, target)
        data = apply_fourier_transform(data, target)
    if datetime_col:
        data = process_datetime(data, datetime_col, format_str)
    if label_encoding:
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
    df['day_of_week'] = df[datetime_col].dt.dayofweek.astype('category')
    df['day_of_week'] = df['day_of_week'].map({
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }).astype('category')
    df['Workday_'] = df['day_of_week'].map({
        'Monday': 'work',
        'Tuesday': 'work',
        'Wednesday': 'work',
        'Thursday': 'work',
        'Friday': 'work',
        'Saturday': 'weekend',
        'Sunday': 'weekend',
    }).astype('category')
    df['Quarter'] = df[datetime_col].dt.month.map({
        1: 'Q1',
        2: 'Q1',
        3: 'Q1',
        4: 'Q2',
        5: 'Q2',
        6: 'Q2',
        7: 'Q3',
        8: 'Q3',
        9: 'Q3',
        10: 'Q4',
        11: 'Q4',
        12: 'Q4',
    }).astype('category')
    df['Month_Week'] = df[datetime_col].dt.day.map({
        1: 'W1',
        2: 'W1',
        3: 'W1',
        4: 'W1',
        5: 'W1',
        6: 'W1',
        7: 'W1',
        8: 'W2',
        9: 'W2',
        10: 'W2',
        11: 'W2',
        12: 'W2',
        13: 'W2',
        14: 'W2',
        15: 'W3',
        16: 'W3',
        17: 'W3',
        18: 'W3',
        19: 'W3',
        20: 'W3',
        21: 'W3',
        22: 'W4',
        23: 'W4',
        24: 'W4',
        25: 'W4',
        26: 'W4',
        27: 'W4',
        28: 'W4',
        29: 'W4',
        30: 'W4',
        31: 'W4',
    }).astype('category')
    df['DayHour'] = df[datetime_col].dt.hour.map({
        0: '7midnight',
        1: '7midnight',
        2: '7midnight',
        3: '7midnight',
        4: '1early-morning',
        5: '1early-morning',
        6: '1early-morning',
        7: '2morning',
        8: '2morning',
        9: '2morning',
        10: '3noon',
        11: '3noon',
        12: '4afternoon',
        13: '4afternoon',
        14: '4afternoon',
        15: '4afternoon',
        16: '4afternoon',
        17: '5evening',
        18: '5evening',
        19: '5evening',
        20: '5evening',
        21: '6night',
        22: '6night',
        23: '6night',
    }).astype('category')
    return df
