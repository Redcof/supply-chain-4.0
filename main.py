import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from logger import logger
from models.dataset import get_dataset
from models.models import get_model
from models.plots import plots
from models.preprocess import timeseries_split


def evaluate(model_name, dataset_name, phase, model, x, y, x_timeseries):
    logger.info(f"Creating dataset for {dataset_name}")
    y_pred = model.predict(x)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred)
    logger.info(f"{dataset_name}: {model_name}:{phase} errors: {dict(mse=mse, rmse=rmse, mae=mae, mape=mape)}")
    # Displaying feature importance scores
    plots("%s_%s_%s" % (dataset_name, model_name, phase), y, y_pred, x_timeseries)


def experiment(model_name, dataset_name):
    logger.info(f"{dataset_name}:{model_name}: Preparing model and datasets")
    data, target, timeseries_col, dataset_name = get_dataset(dataset_name)
    logger.info(f"{dataset_name}:DF INFO:\n{data.info()}")
    model = get_model(model_name)
    x_train, x_test, y_train, y_test = timeseries_split(data, target)
    logger.info(f"{dataset_name}:{model_name}: Creating timeseries features for plotting")
    if timeseries_col:
        x_train_timeseries = x_train[timeseries_col]
        x_test_timeseries = x_test[timeseries_col]
        logger.info(f"{dataset_name}:{model_name}: Dropping datetime before training")
        x_train.drop(columns=[timeseries_col], inplace=True)
        x_test.drop(columns=[timeseries_col], inplace=True)
    else:
        x_train_timeseries = list(range(len(x_train)))
        x_test_timeseries = list(range(len(x_test)))
    logger.info(f"{dataset_name}:{model_name}: Start training...")
    model, feature_importances = model.fit(x_train, y_train)

    logger.info(f"{model_name}:{dataset_name}: feature_importance {feature_importances}")

    logger.info(f"{dataset_name}:{model_name}: Model evaluating")
    evaluate(model_name, dataset_name, "Train", model, x_train, y_train, x_train_timeseries)
    evaluate(model_name, dataset_name, "Test", model, x_test, y_test, x_test_timeseries)
    logger.info(f"{dataset_name}:{model_name}: Done")


def main():
    for model_name in ["xgboost"]:
        for dataset_name in ["product_demand", "food_demand", "livestock_meat_import", "online_retail",
                             "online_retail_2"]:
            experiment(model_name, dataset_name)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)  # logger
    main()
