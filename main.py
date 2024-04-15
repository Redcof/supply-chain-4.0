import os

import mlflow
from dotenv import load_dotenv

load_dotenv("./.env")
import logging

from logger import configure_logger
from tools.dataset import get_dataset
from tools.models import get_model_trainer
from tools.preprocess import timeseries_split

logger = logging.getLogger("SCM-4.0")

is_extra_feature_enabled = True
ablation = 1024 * 5  # set to -1 to select entire dataset, otherwise an integer number


def experiment(model_name, dataset_name):
    logger.info(f"{dataset_name}:{model_name}: Preparing model and datasets")
    df, target, timeseries_col, dataset_name = get_dataset(dataset_name, ablation_limit=ablation,
                                                           is_extra_feature_enabled=is_extra_feature_enabled)
    logger.info(f"{dataset_name}:DF INFO:\n{df.info()}")
    model_trainer = get_model_trainer(model_name)
    x_train, x_test, y_train, y_test = timeseries_split(df, target, train_size=.8)
    # mlflow.log_input(x_train, context="training")
    # mlflow.log_input(x_test, context="testing")
    mlflow.log_params(dict(
        train_size=len(x_train),
        test_size=len(x_test),
        features=x_train.columns.values,
        feature_count=len(x_train.columns)
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
    logger.info(f"{dataset_name}:{model_name}: Start training... WITH DATASIZE: {x_train.shape}")
    model, feature_importance = model_trainer.fit(x_train, y_train, x_test, y_test)

    logger.info(f"{model_name}:{dataset_name}: feature_importance {feature_importance}")

    logger.info(f"{dataset_name}:{model_name}: Start evaluation... WITH DATASIZE: {x_test.shape}")
    model_trainer.evaluate(model_name, dataset_name, "train", model, x_train, y_train, x_train_timeseries)
    model_trainer.evaluate(model_name, dataset_name, "test", model, x_test, y_test, x_test_timeseries)
    logger.info(f"{dataset_name}:{model_name}: Done")


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    for model_name in ["ssl+tabnet", "tabnet"]:
        for dataset_name in ["product_demand", "food_demand", "livestock_meat_import", "online_retail",
                             "online_retail_2"][:1]:
            postfix = "-exf" if is_extra_feature_enabled else ""
            ablation_txt = "-abl" if ablation > 0 else ""
            exp_name = f"{dataset_name}{postfix}{ablation_txt}"
            # exp_name = f"plot-{dataset_name}{postfix}{ablation_txt}"
            mlflow.set_experiment(experiment_name=exp_name)
            with mlflow.start_run(description=exp_name):
                mlflow.log_params(dict(
                    model=model_name,
                ))
                experiment(model_name, dataset_name)
                mlflow.end_run()
    logger.info("All experiments are done")


if __name__ == '__main__':
    configure_logger(logging.DEBUG)  # logger
    main()
