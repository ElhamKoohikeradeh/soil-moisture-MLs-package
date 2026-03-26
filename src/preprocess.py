from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def coerce_numeric(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    df = df.copy()
    cols = columns if columns is not None else df.columns
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def interpolate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    return df

def replace_inf_and_dropna(df: pd.DataFrame, required_columns=None) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=required_columns) if required_columns else df.dropna()

def remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper) | df[col].isna()
    return df.loc[mask].copy()

def prepare_ml_table(df: pd.DataFrame, feature_columns: list[str], target_column: str, remove_outliers=False) -> pd.DataFrame:
    required = [c for c in feature_columns + [target_column] if c in df.columns]
    if target_column not in required:
        raise ValueError(f"Target column '{target_column}' not found.")
    out = df[required].copy()
    out = coerce_numeric(out)
    out = interpolate_numeric(out)
    out = replace_inf_and_dropna(out, required)
    if remove_outliers:
        out = remove_outliers_iqr(out, required)
    return out

def split_xy(df: pd.DataFrame, feature_columns: list[str], target_column: str, test_size=0.2, random_state=42):
    X = df[[c for c in feature_columns if c in df.columns]].copy()
    y = df[target_column].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def align_xy(X: pd.DataFrame, y):
    if isinstance(y, pd.Series):
        y = y.to_frame()
    return X.align(y, join="inner", axis=0)

def resample_xy(X: pd.DataFrame, y, strategy="upsample"):
    if isinstance(y, pd.Series):
        y = y.to_frame()
    data = pd.concat([X, y], axis=1)
    t = y.columns[0]
    majority = data[data[t] == data[t].mode()[0]]
    minority = data[data[t] != data[t].mode()[0]]
    if minority.empty or majority.empty:
        return X.copy(), y.copy()
    if strategy == "upsample":
        minority_res = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        data_res = pd.concat([majority, minority_res], axis=0)
    elif strategy == "downsample":
        majority_res = resample(majority, replace=False, n_samples=len(minority), random_state=42)
        data_res = pd.concat([majority_res, minority], axis=0)
    else:
        raise ValueError("strategy must be 'upsample' or 'downsample'")
    return data_res[X.columns].copy(), data_res[y.columns].copy()

def van_genuchten(theta_r: float, theta_s: float, alpha: float, n: float, psi):
    psi_abs = np.abs(psi)
    m = 1 - 1 / n
    return theta_r + (theta_s - theta_r) / (1 + (alpha * psi_abs) ** n) ** m

def add_vsm_from_potential(df: pd.DataFrame, potential_columns: list[str], theta_r=0.053, theta_s=0.48, alpha=0.195, n=2.13, suffix="_VSM") -> pd.DataFrame:
    out = df.copy()
    for col in potential_columns:
        if col in out.columns:
            out[f"{col}{suffix}"] = van_genuchten(theta_r, theta_s, alpha, n, out[col].values)
    return out
