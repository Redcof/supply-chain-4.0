# Hyperparameter tuning using grid search
import logging

import mlflow
import numpy as np
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_percentage_error

from models import BaseModel

logger = logging.getLogger(__name__)


# reference: https://medium.com/@vanillaxiangshuyang/self-supervised-learning-on-tabular-data-with-tabnet-544b3ec85cee
class MAPE(Metric):
    """
    Mean Absolute Percentage Error.
    """

    def __init__(self):
        self._name = "mape"
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute MAPE (Mean Absolute Percentage Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            RMSE of predictions vs targets.
        """
        return mean_absolute_percentage_error(y_true, y_score)


class TabNetBase(BaseModel):
    @staticmethod
    def prepare_data(x, y):
        x = x.fillna(0)
        np_x = x.to_numpy()
        np_y = y.to_numpy().reshape((-1, 1))
        return np_x, np_y


class TabNetSSL(TabNetBase):
    def __init__(self, fraction=.3, masking_ratio=.8):
        super().__init__()
        self.model_name = "ssl+tabnet"
        self.fraction = fraction
        self.masking_ratio = masking_ratio

    def fit(self, x_train, y_train, x_test, y_test):
        np_x_train, _ = self.prepare_data(x_train, y_train)
        np_x_test, _ = self.prepare_data(x_test, y_test)
        # run pre-training on the entire train-set
        pre_trainer = TabNetPretrainer(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=0.02),
            mask_type="sparsemax",
            seed=47
        )

        pre_trainer.fit(
            max_epochs=80,
            X_train=np_x_train,
            eval_set=[np_x_train, np_x_test],
            eval_name=["train", "test"],
            pretraining_ratio=self.masking_ratio,
        )

        # get a fraction of train data to finetune
        data_l = len(x_train)
        part_of_x_train = x_train[int(data_l * self.fraction):]
        part_of_y_train = y_train[int(data_l * self.fraction):]
        tn = TabNetwork()
        return tn.fit(part_of_x_train, part_of_y_train, x_test, y_test, pre_trained_backbone=pre_trainer)

    def evaluate(self, model_name, dataset_name, phase, model, x, y, x_timeseries, meta_info=""):
        x = x.fillna(0)
        np_x = x.to_numpy()
        y_pred = np.clip(model.predict(np_x), a_min=0, a_max=None)
        self.log_evaluation(model_name, dataset_name, phase, y, y_pred.reshape((-1,)), x_timeseries)


class TabNetwork(TabNetBase):
    def __init__(self):
        super().__init__()
        self.model_name = "tabnet"

    def fit(self, x_train, y_train, x_test, y_test, pre_trained_backbone=None):
        np_x_train, np_y_train = self.prepare_data(x_train, y_train)
        np_x_test, np_y_test = self.prepare_data(x_test, y_test)

        model = TabNetRegressor(seed=47)
        if pre_trained_backbone:
            model.fit(np_x_train, np_y_train,
                      max_epochs=50,
                      eval_set=[(np_x_train, np_y_train),
                                (np_x_test, np_y_test)],
                      eval_name=["train", "test"],
                      eval_metric=["mse", MAPE, "mae", "rmse"],
                      from_unsupervised=pre_trained_backbone)
        else:
            model.fit(np_x_train, np_y_train,
                      max_epochs=100,
                      eval_set=[(np_x_train, np_y_train),
                                (np_x_test, np_y_test)],
                      eval_name=["train", "test"],
                      eval_metric=["mse", MAPE, "mae", "rmse"], )
        imp = {f"feature_{k}": v for k, v in zip(x_train.columns, model.feature_importances_)}
        mlflow.log_metrics(imp)
        return model, imp

    def evaluate(self, model_name, dataset_name, phase, model, x, y, x_timeseries, meta_info=""):
        x = x.fillna(0)
        np_x = x.to_numpy()
        y_pred = np.clip(model.predict(np_x), a_min=0, a_max=None)
        self.log_evaluation(model_name, dataset_name, phase, y, y_pred.reshape((-1,)), x_timeseries, meta_info)
