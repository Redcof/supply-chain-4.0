# Hyperparameter tuning using grid search
import logging

import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import GridSearchCV

from models import BaseModel

logger = logging.getLogger(__name__)


class EXBModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "explainable_boosting"

    @staticmethod
    def fit(x_train, y_train):
        # param_grid = {
        #     'learning_rate': [0.01, 0.05, 0.005],
        #     'min_samples_leaf': [2, 3],
        #     'max_leaves': [3, 5],
        # }
        model = ExplainableBoostingRegressor(random_state=47)
        # grid_search = GridSearchCV(model, param_grid, n_jobs=3, verbose=3)
        # grid_search_agent = grid_search.fit(x_train, y_train)
        # best_model = grid_search_agent.best_estimator_
        model.fit(x_train, y_train)
        global_explanation = model.explain_global()
        # print(global_explanation.selector.head(100))
        # from interpret import show
        # show(global_explanation)
        # plt.show(block=True)
        # input()
        return model, global_explanation
