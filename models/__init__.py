import logging
from abc import abstractmethod, ABC

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from tools.plots import plots

logger = logging.getLogger(__name__)


class BaseModel(ABC):

    @abstractmethod
    def fit(self, x_train, y_train, x_test, y_test):
        ...

    def __init__(self, *args):
        ...

    def evaluate(self, model_name, dataset_name, phase, model, x, y, x_timeseries):
        logger.info(f"predicting {dataset_name}:{model_name}")
        y_pred = model.predict(x)
        self.log_evaluation(model_name, dataset_name, phase, y, y_pred, x_timeseries)

    @staticmethod
    def log_evaluation(model_name, dataset_name, phase, y, y_pred, x_timeseries):
        logger.info(f"evaluating {dataset_name}:{model_name}")
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
        # run_id = mlflow.active_run()
        # for y_, y_h in zip(y, y_pred):
        #     mlflow.log_metrics({f"{phase}_ground_truth": y_, f"{phase}_predicted": y_h})
        # Displaying feature importance scores
        plots("%s_%s_%s" % (dataset_name, model_name, phase), y, y_pred, x_timeseries)
