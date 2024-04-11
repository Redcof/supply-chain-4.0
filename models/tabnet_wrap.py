# Hyperparameter tuning using grid search
import logging

import mlflow
import numpy as np
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor

from models import BaseModel

logger = logging.getLogger(__name__)


class TabNetSSL(BaseModel):
    def __init__(self, fraction=.3):
        super().__init__()
        self.model_name = "ssl+tabnet"
        self.fraction = fraction

    @staticmethod
    def fit(x_train, y_train):
        x_train = x_train.fillna(0)
        data_l = len(x_train)
        fraction = .3
        part_of_x_train = x_train[int(data_l * fraction):]
        part_of_y_train = y_train[int(data_l * fraction):]

        unsupervised_model = TabNetPretrainer(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type="sparsemax",
            seed=47
        )

        unsupervised_model.fit(
            X_train=x_train.to_numpy(),
            pretraining_ratio=0.8,
        )

        tn = TabNetwork()
        return tn.fit(part_of_x_train, part_of_y_train, unsupervised_model)

    def evaluate(self, model_name, dataset_name, phase, model, x, y, x_timeseries):
        x = x.fillna(0)
        np_x = x.to_numpy()
        y_pred = np.clip(model.predict(np_x), a_min=0, a_max=None)
        self.log_evaluation(model_name, dataset_name, phase, y, y_pred.reshape((-1,)), x_timeseries)


class TabNetwork(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "tabnet"

    @staticmethod
    def fit(x_train, y_train, ssl_model=None):
        x_train = x_train.fillna(0)
        np_x_train = x_train.to_numpy()
        np_y_train = y_train.to_numpy().reshape((-1, 1))
        model = TabNetRegressor(seed=47)
        if ssl_model:
            model.fit(np_x_train, np_y_train, from_unsupervised=ssl_model)
        else:
            model.fit(np_x_train, np_y_train)
        imp = {f"feature_{k}": v for k, v in zip(x_train.columns, model.feature_importances_)}
        mlflow.log_params(imp)
        return model, imp

    def evaluate(self, model_name, dataset_name, phase, model, x, y, x_timeseries):
        x = x.fillna(0)
        np_x = x.to_numpy()
        y_pred = np.clip(model.predict(np_x), a_min=0, a_max=None)
        self.log_evaluation(model_name, dataset_name, phase, y, y_pred.reshape((-1,)), x_timeseries)
