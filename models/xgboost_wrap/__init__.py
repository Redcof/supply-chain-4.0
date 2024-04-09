# Hyperparameter tuning using grid search
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from models import timeseries
from models.plots import plots


def trainer(data, target):
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'reg_alpha': [0.1],
        'reg_lambda': [0.1],
    }

    X_train, X_test, y_train, y_test = timeseries.timeseries_split(data, target)
    xgb_model = XGBRegressor(eval_metric=["rmse", "mae", "mape"], enable_categorical=True, random_state=47)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3)

    z = grid_search.fit(X_train, y_train)

    # Evaluating the XGBoost model on the testing set
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    xgb_model = z.best_estimator_
    y_pred = xgb_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(dict(mse=mse, rmse=rmse, mae=mae, mape=mape))
    # Displaying feature importance scores
    print("feature_importance", xgb_model.feature_importances_)
    df = pd.DataFrame(dict(y=y_test, y_pred=y_pred))
    plots(y_test, y_pred)
