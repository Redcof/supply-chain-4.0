# Hyperparameter tuning using grid search
import logging

from interpret.glassbox import ExplainableBoostingRegressor

from models import BaseModel

logger = logging.getLogger(__name__)


class EXBModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "explainable_boosting"

    def fit(self, x_train, y_train, x_test, y_test):
        model = ExplainableBoostingRegressor(random_state=47)
        model.fit(x_train, y_train)
        global_explanation = model.explain_global()
        plotly_fig = global_explanation.visualize()
        
        return model, plotly_fig
