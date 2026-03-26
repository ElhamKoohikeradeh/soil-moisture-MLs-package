from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from .models import build_model_specs, maybe_grid_search

def train_and_evaluate(X_train, X_test, y_train, y_test, random_state=42, cv=10):
    specs = build_model_specs(random_state=random_state)
    metrics_rows = []
    predictions = {}
    fitted_models = {}
    for model_name, spec in specs.items():
        model = maybe_grid_search(spec, cv=cv, scoring="r2")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        fitted_models[model_name] = model
        predictions[model_name] = y_pred
        metrics_rows.append({"Model": model_name, "MSE": mse, "R2": r2})
    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(predictions, index=y_test.index)
    predictions_df["Observed"] = y_test.values
    return {"metrics": metrics_df, "predictions": predictions_df}, fitted_models

def cross_validate_models(X, y, random_state=42, cv=10):
    specs = build_model_specs(random_state=random_state)
    rows = []
    for model_name, spec in specs.items():
        estimator = spec["estimator"]
        scores = cross_val_score(estimator, X, y.values.ravel() if hasattr(y, "values") else y, cv=cv, scoring="r2")
        rows.append({"Model": model_name, "CV_R2_Mean": float(np.mean(scores)), "CV_R2_Std": float(np.std(scores))})
    return pd.DataFrame(rows)

def save_training_outputs(output_dir, results, fitted_models):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results["metrics"].to_csv(output_dir / "ML_metrics.csv", index=False)
    results["predictions"].to_csv(output_dir / "ML_predictions.csv", index=False)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for name, model in fitted_models.items():
        joblib.dump(model, models_dir / f"{name.lower()}_model.pkl")
