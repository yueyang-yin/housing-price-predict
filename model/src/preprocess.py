from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    drop_first: bool = True  # Align with notebook's get_dummies(drop_first=True)
    target_col: str = "price_log"
    raw_price_col: str = "price"


def _split_statezip(df: pd.DataFrame) -> pd.DataFrame:
    """Split statezip into state + zipcode."""
    if "statezip" in df.columns:
        sz = df["statezip"].astype(str).str.strip()
        df["state"] = sz.str.split().str[0]
        df["zipcode"] = sz.str.split().str[-1]
    return df


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Log transforms (training may include price; inference usually won't)
    if "price" in df.columns and "price_log" not in df.columns:
        df["price_log"] = np.log(df["price"].astype(float))

    if "sqft_lot" in df.columns:
        df["sqft_lot_log"] = np.log1p(df["sqft_lot"].astype(float))

    # Parse date into year/month, then drop original date (avoid object dtype)
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["year_sold"] = dt.dt.year
        df["month_sold"] = dt.dt.month
        df = df.drop(columns=["date"])

    if "yr_renovated" in df.columns:
        df["is_renovated"] = (df["yr_renovated"].fillna(0).astype(float) > 0).astype(int)

    if "year_sold" in df.columns and "yr_built" in df.columns:
        df["house_age"] = df["year_sold"].astype(float) - df["yr_built"].astype(float)

    if "sqft_above" in df.columns and "sqft_living" in df.columns:
        df["above_ratio"] = df["sqft_above"].astype(float) / df["sqft_living"].astype(float)

    if "sqft_basement" in df.columns and "sqft_living" in df.columns:
        df["basement_ratio"] = df["sqft_basement"].astype(float) / df["sqft_living"].astype(float)

    if "bathrooms" in df.columns and "bedrooms" in df.columns:
        # bedrooms=0 can produce inf; handled later
        df["bath_ratio"] = df["bathrooms"].astype(float) / df["bedrooms"].astype(float)

    for c in ["street", "country"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    if "yr_built" in df.columns:
        df = df.drop(columns=["yr_built"])

    return df


def raw_to_model_ready(
    raw_df: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
) -> pd.DataFrame:
    """
    Convert raw rows into model-ready features (with one-hot encoding).
    """
    cfg = config or PreprocessConfig()
    df = raw_df.copy()

    df = _split_statezip(df)
    df = _add_basic_features(df)

    # Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # One-hot encode location fields if present
    cat_cols = [c for c in ["city", "state", "zipcode"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=cfg.drop_first)

    return df


def split_xy(model_ready_df: pd.DataFrame, config: Optional[PreprocessConfig] = None):
    """Split model_ready into X and y (if present); keep X numeric-only."""
    cfg = config or PreprocessConfig()
    df = model_ready_df.copy()

    y = None
    if cfg.target_col in df.columns:
        y = df[cfg.target_col].astype(float)

    X = df.drop(columns=[c for c in [cfg.raw_price_col, cfg.target_col] if c in df.columns], errors="ignore")

    # XGBoost requires numeric/bool dtypes
    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    return X, y


def align_to_feature_columns(X: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Align columns to training-time feature list (missing -> 0)."""
    X_aligned = X.reindex(columns=feature_columns, fill_value=0)
    return X_aligned


def save_feature_columns(path: str, feature_columns: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)


def load_feature_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)