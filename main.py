import os

import mlflow
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv("./.env")
import logging

from logger import configure_logger
from tools.dataset import get_dataset
from tools.models import get_model_trainer
from tools.preprocess import timeseries_split

logger = logging.getLogger("SCM-4.0")

# is_extra_feature_enabled = False
ablation = -1  # 1024 * 5  # set to -1 to select entire dataset, otherwise an integer number

unique_mlops_exp_prefix = "macebm"


def calculate_time_period(series):
    difference_txt = ""
    # Get the difference between the maximum and minimum dates
    date_difference = series.max() - series.min()

    # # Extract individual components
    # years = date_difference.days // 365
    # months = (date_difference.days % 365) // 30
    # days = (date_difference.days % 365) % 30
    # hours = date_difference.seconds // 3600
    # minutes = (date_difference.seconds % 3600) // 60
    # seconds = date_difference.seconds % 60
    # for unit, val in zip(("Y", "M", "D", "H", "m", "s"), (years, months, days, hours, minutes, seconds)):
    #     if val > 0 or unit in ("H", "m", "s"):
    #         difference_txt = f"{difference_txt} {val}{unit}"
    return date_difference.seconds, str(date_difference)  # , difference_txt.strip()


def experiment(model_name, dataset_name, is_extra_feature_enabled, extra_feat_txt="", ablation_txt=""):
    meta_info = "%s%s" % (extra_feat_txt, ablation_txt)
    logger.info(f"{dataset_name}:{model_name}:{meta_info} Preparing model and datasets")
    df, target, timeseries_col, dataset_name = get_dataset(dataset_name, ablation_limit=ablation,
                                                           is_extra_feature_enabled=is_extra_feature_enabled)
    logger.info(f"{dataset_name}:DF INFO:\n{df.info()}")
    model_trainer = get_model_trainer(model_name)
    x_train, x_test, y_train, y_test = timeseries_split(df, target, train_size=.8)

    total_diff = calculate_time_period(df[timeseries_col])
    train_diff = calculate_time_period(x_train[timeseries_col])
    test_diff = calculate_time_period(x_test[timeseries_col])
    mlflow.log_params(dict(
        train_size=len(x_train),
        test_size=len(x_test),
        feature_count=len(x_train.columns),
        time_period=total_diff[0],
        time_period_sec_train=train_diff[0],
        time_period_sec_test=test_diff[0],
        features=list(x_train.columns.values),
        time_period_sec_txt=total_diff[1],
        time_period_train_txt=train_diff[1],
        time_period_test_txt=test_diff[1],
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
    if model_name == "explainable_boosting":
        feature_importance.write_html(f"output/{dataset_name}-{model_name}{meta_info}.html")
        feature_importance = "are saved as plotly html."
    logger.info(f"{model_name}:{dataset_name}: feature_importance {feature_importance}")

    logger.info(f"{dataset_name}:{model_name}: Start evaluation... WITH DATASIZE: {x_test.shape}")
    model_trainer.evaluate(model_name, dataset_name, f"train", model, x_train, y_train, x_train_timeseries,
                           meta_info=meta_info)
    model_trainer.evaluate(model_name, dataset_name, f"test", model, x_test, y_test, x_test_timeseries,
                           meta_info=meta_info)
    logger.info(f"{dataset_name}:{model_name}:{meta_info} Done")
    return True


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiments = []
    # global is_extra_feature_enabled

    for dataset_name in ["online_retail", "online_retail_2", "product_demand", "livestock_meat_import",
                         "future_sales", ]:
        for model_name in ["explainable_boosting"]:
            experiments.append((model_name, dataset_name))
    print("")

    for is_extra_feature_enabled in [False, True]:
        for model_name, dataset_name in tqdm(experiments):
            extra_feat_txt = "-exf" if is_extra_feature_enabled else ""
            ablation_txt = f"-abl{ablation}" if ablation > 0 else ""

            exp_name = f"{unique_mlops_exp_prefix}-{dataset_name}{extra_feat_txt}{ablation_txt}"
            experiment_tracking_file = f"output/tracking/{dataset_name}-{model_name}{extra_feat_txt}{ablation_txt}"

            if not os.path.exists(experiment_tracking_file):
                logger.info("%s [Executing...]" % experiment_tracking_file)
                mlflow.set_experiment(experiment_name=exp_name)
                with mlflow.start_run(description=exp_name):
                    mlflow.log_params(dict(
                        model=model_name,
                        dataset=dataset_name,
                        extra_feat=is_extra_feature_enabled,
                        ablation=ablation,
                    ))
                    experiment(model_name, dataset_name, is_extra_feature_enabled, extra_feat_txt, ablation_txt)
                    mlflow.end_run()
                    open(experiment_tracking_file, "w")
            else:
                logger.info("%s [DONE...]" % experiment_tracking_file)

    logger.info("All experiments are done")


if __name__ == '__main__':
    configure_logger(logging.DEBUG)  # logger
    # dataset_name = "product_demand"
    # df, target, timeseries_col, dataset_name = get_dataset(dataset_name, ablation_limit=-1,
    # is_extra_feature_enabled=False,label_encoding=False)
    main()
