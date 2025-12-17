from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from preprocess import (
    raw_to_model_ready,
    split_xy,
    load_feature_columns,
    align_to_feature_columns,
)


def load_model(artifacts_dir: Path) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(str(artifacts_dir / "xgb_model.json"))
    return model


def predict_from_model_ready_df(model: XGBRegressor, model_ready_df: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    feature_columns = load_feature_columns(str(artifacts_dir / "feature_columns.json"))
    X, _ = split_xy(model_ready_df)

    X = align_to_feature_columns(X, feature_columns)

    pred_log = model.predict(X)
    pred_price = np.exp(pred_log)

    out = pd.DataFrame({
        "pred_price_log": pred_log,
        "pred_price": pred_price
    })
    return out


def predict_from_raw_df(model: XGBRegressor, raw_df: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    # Raw -> model-ready features
    model_ready = raw_to_model_ready(raw_df)
    return predict_from_model_ready_df(model, model_ready, artifacts_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV (one or many rows).")
    parser.add_argument("--mode", type=str, choices=["raw", "model_ready"], default="raw",
                        help="raw=original fields; model_ready=one-hot feature table.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "model" / "artifacts"

    if not (artifacts_dir / "xgb_model.json").exists():
        raise FileNotFoundError("Trained model not found. Run: python model/src/train.py")

    df = pd.read_csv(args.input)

    model = load_model(artifacts_dir)

    if args.mode == "raw":
        out = predict_from_raw_df(model, df, artifacts_dir)
    else:
        out = predict_from_model_ready_df(model, df, artifacts_dir)

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()