import os

import mlflow
from dotenv import load_dotenv

load_dotenv("./.env")
import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from logger import configure_logger
from tools.dataset import get_dataset
from tools.models import get_model
from tools.plots import plots
from tools.preprocess import timeseries_split

logger = logging.getLogger("SCM-4.0")

is_extra_feature_enabled = True
ablation = 1000


def evaluate(model_name, dataset_name, phase, model, x, y, x_timeseries):
    logger.info(f"Creating dataset for {dataset_name}")
    y_pred = model.predict(x)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred)
    logger.info(f"{dataset_name}: {model_name}:{phase} errors: {dict(mse=mse, rmse=rmse, mae=mae, mape=mape)}")
    mlflow.log_params({
        f"{phase}_mae": mae,
        f"{phase}_mse": mse,
        f"{phase}_rmse": rmse,
        f"{phase}_mape": mape,
    })
    for y_, y_h in zip(y, y_pred):
        mlflow.log_metrics({f"{phase}_ground_truth": y_, f"{phase}_predicted": y_h})
    # Displaying feature importance scores
    plots("%s_%s_%s" % (dataset_name, model_name, phase), y, y_pred, x_timeseries)


def experiment(model_name, dataset_name):
    logger.info(f"{dataset_name}:{model_name}: Preparing model and datasets")
    data, target, timeseries_col, dataset_name = get_dataset(dataset_name, ablation_limit=ablation,
                                                             is_extra_feature_enabled=is_extra_feature_enabled)
    logger.info(f"{dataset_name}:DF INFO:\n{data.info()}")
    model = get_model(model_name)
    x_train, x_test, y_train, y_test = timeseries_split(data, target)
    mlflow.log_params(dict(
        train_size=len(x_train),
        test_size=len(x_test),
        features=x_train.columns
    ))
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
    logger.info(f"{dataset_name}:{model_name}: Start training... WITH DATASIZE: {data.shape}")
    model, feature_importances = model.fit(x_train, y_train)

    logger.info(f"{model_name}:{dataset_name}: feature_importance {feature_importances}")

    logger.info(f"{dataset_name}:{model_name}: Model evaluating")
    evaluate(model_name, dataset_name, "Train", model, x_train, y_train, x_train_timeseries)
    evaluate(model_name, dataset_name, "Test", model, x_test, y_test, x_test_timeseries)
    logger.info(f"{dataset_name}:{model_name}: Done")


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    for model_name in ["explainable_boosting"]:
        for dataset_name in ["product_demand", "food_demand", "livestock_meat_import", "online_retail",
                             "online_retail_2"]:
            postfix = "-extra_features" if is_extra_feature_enabled else ""
            ablation_txt = "-ablation" if ablation > 0 else ""
            exp_name = f"{model_name}-{dataset_name}{postfix}{ablation_txt}"
            mlflow.set_experiment(experiment_name=exp_name)
            mlflow.start_run(description=exp_name)
            mlflow.log_params(dict(
                dataset=dataset_name,
                model=model_name,
                extra_features=is_extra_feature_enabled,
            ))
            experiment(model_name, dataset_name)
            mlflow.end_run()


if __name__ == '__main__':
    configure_logger(logging.DEBUG)  # logger
    main()
