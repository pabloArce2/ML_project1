
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MODEL_DIR = RESULTS_DIR / "majority_baseline_classification"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "model.joblib"
DEFAULT_TARGETS_PATH = DEFAULT_MODEL_DIR / "targets.json"

def load_artifacts(model_dir: Path) -> tuple[int, str]:
    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    targets_path = model_dir / "targets.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets file not found: {targets_path}")

    md = load(model_path)
    # model saved as {"type": "majority_baseline", "target": "chd", "majority_class": int, "majority_frac": float}
    majority_class = int(md["majority_class"])
    targets = json.loads(targets_path.read_text())
    if len(targets) != 1:
        raise ValueError(f"Expected a single target in targets.json, got: {targets}")
    target = targets[0]
    return majority_class, target

def predict_majority(df: pd.DataFrame, majority_class: int, target: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{target}_pred"] = majority_class
    return out

def maybe_report_accuracy(df_with_preds: pd.DataFrame, target: str) -> None:
    if target in df_with_preds.columns:
        y_true = df_with_preds[target].to_numpy()
        y_pred = df_with_preds[f"{target}_pred"].to_numpy()
        # Ensure ints
        y_true = y_true.astype(int, copy=False)
        y_pred = y_pred.astype(int, copy=False)
        acc = float(np.mean(y_true == y_pred))
        print(f"[info] Accuracy on provided data (since '{target}' is present): {acc:.4f}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for majority-class baseline classifier (predicts most frequent class).")
    p.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR),
                   help=f"Directory containing model.joblib and targets.json (default: {DEFAULT_MODEL_DIR})")
    p.add_argument("--input", type=str, required=True, help="Path to input CSV with features (and optionally the true 'chd').")
    p.add_argument("--output", type=str, required=True, help="Where to write the CSV with predictions appended.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    majority_class, target = load_artifacts(Path(args.model_dir))

    df = pd.read_csv(args.input)
    out = predict_majority(df, majority_class=majority_class, target=target)

    maybe_report_accuracy(out, target=target)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[ok] Wrote predictions to: {out_path}")
    print(f"[ok] Majority class used: {majority_class} (target='{target}')")

if __name__ == "__main__":
    main()
