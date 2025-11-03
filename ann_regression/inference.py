# ann_regression/inference.py
"""
Load a saved ANN regression model and run predictions on a CSV of (raw or engineered) features.

Expected artefacts (produced by ann_regression/run.py → fit_and_save_final_ann):
- results/<setting>/model.joblib
- results/<setting>/feature_names.json   (list of feature column names used for training)
- results/<setting>/targets.json         (list of target names, e.g. ["ldl"])

Usage example:
  python -m ann_regression.inference \
      --setting ldl_only \
      --input path/to/new_samples.csv \
      --output path/to/preds.csv
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
    """Load model + metadata for a given setting (e.g., 'ldl_only')."""
    folder = results_dir / setting
    model_path = folder / "model.joblib"
    feat_path = folder / "feature_names.json"
    targ_path = folder / "targets.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_names.json not found: {feat_path}")
    if not targ_path.exists():
        raise FileNotFoundError(f"targets.json not found: {targ_path}")

    model = load(model_path)
    feature_names: List[str] = json.loads(feat_path.read_text())
    targets: List[str] = json.loads(targ_path.read_text())
    return model, feature_names, targets, folder


def prepare_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Align an arbitrary input DataFrame to the trained feature space:
      1) get_dummies to expand any categoricals (e.g., 'famhist').
      2) add any missing training columns with zeros (baseline level).
      3) drop any extra columns not seen during training.
      4) order columns exactly as in feature_names.
    """
    # expand categoricals to one-hot (consistent with training which used get_dummies(drop_first=True))
    df_enc = pd.get_dummies(df, drop_first=True, dtype=float)

    # ensure every training feature exists
    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0.0

    # keep only training features, in the correct order
    df_enc = df_enc[feature_names]
    return df_enc


def maybe_extract_ground_truth(df: pd.DataFrame, targets: List[str]) -> np.ndarray | None:
    """If the input CSV also contains true target columns, return them for quick scoring."""
    if all(t in df.columns for t in targets):
        y_df = df[targets]
        y = y_df.values if len(targets) > 1 else y_df.values.ravel()
        return y
    return None


def main():
    parser = argparse.ArgumentParser(description="ANN regression inference")
    parser.add_argument("--setting", required=True, help="Target setup name (e.g., 'ldl_only')")
    parser.add_argument("--input", required=True, type=Path, help="CSV file with samples to score")
    parser.add_argument("--output", type=Path, default=None, help="Where to write predictions CSV")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Folder with saved model/metadata")
    args = parser.parse_args()

    model, feature_names, targets, folder = load_artefacts(args.results_dir, args.setting)

    # load input rows
    df_in = pd.read_csv(args.input)

    # keep a copy of original for concatenation
    original_cols = df_in.columns.tolist()

    # prepare features to match training space
    X_enc = prepare_features(df_in, feature_names)

    # predict
    preds = model.predict(X_enc.values)
    if preds.ndim == 1:
        preds = preds[:, np.newaxis]

    preds_df = pd.DataFrame(preds, columns=targets)

    # attach predictions to original input order
    out_df = pd.concat([df_in.reset_index(drop=True), preds_df], axis=1)

    # (optional) quick score if ground truth present
    y_true = maybe_extract_ground_truth(df_in, targets)
    if y_true is not None:
        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        mse = float(np.mean((preds_df.values.ravel() - y_true) ** 2))
        print(f"[info] Detected ground-truth in input. MSE on provided rows: {mse:.4f}")

    # write output
    if args.output is None:
        # default: results/<setting>/inference_predictions.csv
        out_path = folder / "inference_predictions.csv"
    else:
        out_path = args.output
        if out_path.exists() and out_path.is_dir():
            out_path = out_path / "inference_predictions.csv"

    out_df.to_csv(out_path, index=False)
    print(f"[ok] Wrote predictions → {out_path}")


if __name__ == "__main__":
    main()

