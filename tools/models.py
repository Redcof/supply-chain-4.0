import logging

from models.explanable_boosting_wrap import EXBModel
from models.tabnet_wrap import TabNetSSL, TabNetwork
from models.xgboost_wrap import XGBoostBaseModel

logger = logging.getLogger(__name__)

all_models = [
    "xgboost",
    "explainable_boosting",
    "tabnet",
    "ssl+tabnet",
]


def get_model(model_name):
    assert model_name in all_models
    logger.info("Create model trainer object")
    if model_name == "xgboost":
        return XGBoostBaseModel()
    elif model_name == "explainable_boosting":
        return EXBModel()
    elif model_name == "tabnet":
        return TabNetwork()
    elif model_name == "ssl+tabnet":
        return TabNetSSL()
