from __future__ import annotations
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def build_model_specs(random_state: int = 42):
    return {
        "MLR": {"estimator": LinearRegression(), "grid": None},
        "SVR": {"estimator": SVR(kernel="rbf"), "grid": {"C": np.arange(0.1, 1.6, 0.1), "gamma": np.arange(0.1, 1.6, 0.1)}},
        "RFR": {"estimator": RandomForestRegressor(random_state=random_state), "grid": {"n_estimators": [50,100,200,300,400,500], "max_depth": list(range(5,16))}},
        "ABR": {"estimator": AdaBoostRegressor(random_state=random_state), "grid": {"n_estimators": list(range(50,501,50)), "learning_rate": np.arange(0.01, 1.01, 0.01)}},
    }

def maybe_grid_search(spec: dict, cv=10, scoring="r2"):
    est = spec["estimator"]
    grid = spec["grid"]
    return est if not grid else GridSearchCV(est, grid, cv=cv, scoring=scoring, n_jobs=-1)
