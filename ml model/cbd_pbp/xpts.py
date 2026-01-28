from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Optional, Dict, Any, Tuple

def fit_xpts_model(shots: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame]:
    # shots: one row per attempt with columns: zone, range, family, assisted_flag, is_three, made, points
    df = shots.dropna(subset=["made"]).copy()
    # define is_three from family or points
    df["is_three"] = df.get("is_three")
    if "is_three" not in df.columns or df["is_three"].isna().all():
        df["is_three"] = df["family"].astype(str).str.contains("three", case=False, na=False)
    # classification model for make prob
    X = df[["zone","range","assisted_flag","is_three"]].copy()
    y = df["made"].astype(int)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["zone","range"]),
        ("num", "passthrough", ["assisted_flag","is_three"]),
    ])
    model = LogisticRegression(max_iter=200, n_jobs=1)
    pipe = Pipeline([("pre", pre), ("lr", model)])
    pipe.fit(X, y)

    # calibration table: expected points = p_make * (3 if is_three else 2)
    df["p_make"] = pipe.predict_proba(X)[:,1]
    df["xpts"] = df["p_make"] * np.where(df["is_three"], 3.0, 2.0)
    return pipe, df

def predict_xpts(pipe: Pipeline, shots: pd.DataFrame) -> np.ndarray:
    X = shots[["zone","range","assisted_flag","is_three"]].copy()
    p = pipe.predict_proba(X)[:,1]
    return p * np.where(shots["is_three"].astype(bool), 3.0, 2.0)
