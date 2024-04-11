# Hyperparameter tuning using grid search
import logging

from interpret.glassbox import ExplainableBoostingRegressor

from models import BaseModel

logger = logging.getLogger(__name__)


class EXBModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "explainable_boosting"

    @staticmethod
    def fit(x_train, y_train):
        model = ExplainableBoostingRegressor(random_state=47)
        model.fit(x_train, y_train)
        # global_explanation = model.explain_global()
        return model, {}
