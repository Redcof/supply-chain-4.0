import logging

from models.xgboost_wrap import XGBoostBaseModel

logger = logging.getLogger(__file__)


def get_model(model_name):
    logger.info("Create model trainer object")
    if model_name == "xgboost":
        return XGBoostBaseModel()
    else:
        raise ValueError(f"Model: {model_name} is not implemented")
