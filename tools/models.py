import logging

from models.explanable_boosting_wrap import EXBModel
from models.xgboost_wrap import XGBoostBaseModel

logger = logging.getLogger(__name__)


def get_model(model_name):
    logger.info("Create model trainer object")
    if model_name == "xgboost":
        return XGBoostBaseModel()
    elif model_name == "explainable_boosting":
        return EXBModel()
    else:
        raise ValueError(f"Model: {model_name} is not implemented")
