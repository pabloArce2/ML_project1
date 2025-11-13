"""
Load a saved *classification* ANN or logistic model and run predictions on a CSV of raw features.
It aligns columns to the saved feature space (get_dummies + add-missing + order).

Expected artefacts (from run.py --fit-final):
- results/chd_only/model.joblib
- results/chd_only/feature_names.json
- results/chd_only/targets.json  -> ["chd"]
"""

import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from joblib import load

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

def load_artefacts(results_dir: Path, setting: str):
    folder = results_dir / setting
    model = load(folder / "model.joblib")
    feature_names: List[str] = json.loads((folder / "feature_names.json").read_text())
    targets: List[str] = json.loads((folder / "targets.json").read_text())
    return model, feature_names, targets, folder

def prepare_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df_enc = pd.get_dummies(df, drop_first=True, dtype=float)
    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0.0
    df_enc = df_enc[feature_names]
    return df_enc

def main():
    parser = argparse.ArgumentParser(description="Classification inference (ANN/LogReg)")
    parser.add_argument("--setting", default="chd")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    model, feature_names, targets, folder = load_artefacts(args.results_dir, args.setting)
    df_in = pd.read_csv(args.input)
    X = prepare_features(df_in.drop(columns=["chd"], errors="ignore"), feature_names)

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X.values)[:, 1]
    preds = model.predict(X.values)

    out_df = df_in.copy()
    out_df["chd_pred"] = preds
    if probs is not None:
        out_df["chd_prob"] = probs

    if args.output is None:
        out_path = folder / "inference_predictions.csv"
    else:
        out_path = args.output
        if out_path.exists() and out_path.is_dir():
            out_path = out_path / "inference_predictions.csv"

    out_df.to_csv(out_path, index=False)
    print(f"[ok] Wrote predictions → {out_path}")

if __name__ == "__main__":
    main()
