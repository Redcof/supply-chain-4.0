# Hyperparameter tuning using grid search
import logging

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from models import BaseModel

logger = logging.getLogger(__file__)


class XGBoostBaseModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "xgboost"

    @staticmethod
    def fit(x_train, y_train):
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'reg_alpha': [0.1],
            'reg_lambda': [0.1],
        }
        xgb_model = XGBRegressor(eval_metric=["rmse", "mae", "mape"], enable_categorical=True, random_state=47)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3)
        grid_search_agent = grid_search.fit(x_train, y_train)
        xgb_model = grid_search_agent.best_estimator_
        return xgb_model, xgb_model.feature_importances_