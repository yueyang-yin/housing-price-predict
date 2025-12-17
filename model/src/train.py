from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import split_xy, save_feature_columns


def accuracy_within_pct(y_true_price: np.ndarray, y_pred_price: np.ndarray, pct: float = 0.10) -> float:
    """Acc@pct: |pred-true|/true <= pct."""
    y_true_price = np.asarray(y_true_price, dtype=float)
    y_pred_price = np.asarray(y_pred_price, dtype=float)
    eps = 1e-12
    rel_err = np.abs(y_pred_price - y_true_price) / np.maximum(np.abs(y_true_price), eps)
    return float(np.mean(rel_err <= pct))


def load_model_ready(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 55,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """60/20/20 split: train 60%, cv 20%, test 20%."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=seed
    )
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )
    return X_train, X_cv, X_test, y_train, y_cv, y_test


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    seed: int = 55,
) -> XGBRegressor:
    """Train the final XGBoost model with early stopping."""
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        random_state=seed,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_cv, y_cv)],
        verbose=False,
        early_stopping_rounds=50,
    )
    return model


def evaluate(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
    raw_price_available: bool,
    test_price: np.ndarray | None,
) -> Dict[str, Any]:
    """Evaluate in log space; optionally compute Acc@10% in price space."""
    y_pred_log = model.predict(X_test)
    mse = mean_squared_error(y_test_log, y_pred_log)
    j = mse / 2.0
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test_log, y_pred_log))

    metrics: Dict[str, Any] = {
        "test_mse_log": float(mse),
        "test_j_mse_over_2_log": float(j),
        "test_rmse_log": rmse,
        "test_r2_log": r2,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
    }

    if raw_price_available and test_price is not None:
        pred_price = np.exp(y_pred_log)
        acc10 = accuracy_within_pct(test_price, pred_price, pct=0.10)
        metrics["test_acc_within_10pct_price"] = float(acc10)

    return metrics


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "model_ready.csv"
    artifacts_dir = project_root / "model" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_model_ready(str(data_path))
    X, y = split_xy(df)

    if y is None:
        raise ValueError("price_log column not found in model_ready.csv. Make sure your export includes price_log.")

    # Use raw price for Acc@10% if available
    has_price = "price" in df.columns
    price = df["price"].astype(float).to_numpy() if has_price else None

    X_train, X_cv, X_test, y_train, y_cv, y_test = split_data(X, y, seed=55)
    test_price = price[X_test.index] if (has_price and price is not None) else None

    model = train_xgb(X_train, y_train, X_cv, y_cv, seed=55)

    metrics = evaluate(
        model=model,
        X_test=X_test,
        y_test_log=y_test,
        raw_price_available=has_price,
        test_price=test_price,
    )

    # Save artifacts
    model_path = artifacts_dir / "xgb_model.json"
    model.save_model(str(model_path))

    feature_columns = list(X.columns)
    save_feature_columns(str(artifacts_dir / "feature_columns.json"), feature_columns)

    params = model.get_params()
    with open(artifacts_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("✅ Training done.")
    print(f"- Saved model: {model_path}")
    print(f"- Features: {len(feature_columns)} columns")
    print(f"- Metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()