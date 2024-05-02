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

is_extra_feature_enabled = True
ablation = -1  # 1024 * 5  # set to -1 to select entire dataset, otherwise an integer number


def experiment(model_name, dataset_name, extra_feat_txt="", ablation_txt=""):
    meta_info = "%s%s" % (extra_feat_txt, ablation_txt)
    logger.info(f"{dataset_name}:{model_name}:{meta_info} Preparing model and datasets")
    df, target, timeseries_col, dataset_name = get_dataset(dataset_name, ablation_limit=ablation,
                                                           is_extra_feature_enabled=is_extra_feature_enabled)
    logger.info(f"{dataset_name}:DF INFO:\n{df.info()}")
    model_trainer = get_model_trainer(model_name)
    x_train, x_test, y_train, y_test = timeseries_split(df, target, train_size=.8)
    mlflow.log_params(dict(
        features=list(x_train.columns.values),
    ))
    mlflow.log_metrics(dict(
        train_size=len(x_train),
        test_size=len(x_test),
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
    model_trainer.evaluate(model_name, dataset_name, f"train", model, x_train, y_train, x_train_timeseries,
                           meta_info=meta_info)
    model_trainer.evaluate(model_name, dataset_name, f"test", model, x_test, y_test, x_test_timeseries,
                           meta_info=meta_info)
    logger.info(f"{dataset_name}:{model_name}:{meta_info} Done")
    return True


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiments = []
    for dataset_name in ["livestock_meat_import"]:
        for model_name in ["xgboost", "ssl+tabnet", "tabnet", "explainable_boosting"]:
            experiments.append((model_name, dataset_name))
    print("")
    for model_name, dataset_name in tqdm(experiments):
        extra_feat_txt = "-exf" if is_extra_feature_enabled else ""
        ablation_txt = f"-abl{ablation}" if ablation > 0 else ""
        unique_mlops_exp_prefix = "macmay"
        exp_name = f"{unique_mlops_exp_prefix}-{dataset_name}{extra_feat_txt}{ablation_txt}"
        experiment_tracking_file = f"output/tracking/{dataset_name}-{model_name}{extra_feat_txt}{ablation_txt}"
        print(experiment_tracking_file, end="")
        if not os.path.exists(experiment_tracking_file):
            print("[Executing...]")
            mlflow.set_experiment(experiment_name=exp_name)
            with mlflow.start_run(description=exp_name):
                mlflow.log_params(dict(
                    model=model_name,
                    dataset=dataset_name,
                    extra_feat=is_extra_feature_enabled,
                    ablation=ablation,
                ))
                experiment(model_name, dataset_name, extra_feat_txt, ablation_txt)
                mlflow.end_run()
                open(experiment_tracking_file, "w")
        else:
            print("[DONE]")


logger.info("All experiments are done")

if __name__ == '__main__':
    configure_logger(logging.DEBUG)  # logger
    main()
