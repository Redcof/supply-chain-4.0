# Hyperparameter tuning using grid search
import logging

import mlflow
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from models import BaseModel

logger = logging.getLogger(__name__)


class XGBoostBaseModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "xgboost"
    
    def fit(self, x_train, y_train, x_test, y_test):
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'reg_alpha': [0.1],
            'reg_lambda': [0.1],
        }
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        xgb_model = XGBRegressor(enable_categorical=True, random_state=47, device=device)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, n_jobs=3, verbose=1)
        grid_search_agent = grid_search.fit(x_train, y_train)
        xgb_model = grid_search_agent.best_estimator_
        imp = {f"feature_{k}": v for k, v in zip(xgb_model.feature_names_in_, xgb_model.feature_importances_)}
        mlflow.log_params(imp)
        return xgb_model, imp
