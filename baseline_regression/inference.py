"""
Utility script to emit baseline predictions for any provided rows.
Since this baseline ignores features, it only loads the stored mean and
produces a constant column "ldl_pred".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load

from .run import RESULTS_DIR, TARGET_NAME

MODEL_DIR = RESULTS_DIR / "mean_baseline_regression"


def load_mean(model_dir: Path = MODEL_DIR) -> float:
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Missing baseline artefacts. Run: python -m baseline_regression.run")
    data = load(model_path)
    mean = float(data["mean"])
    return mean


def predict_const(n_rows: int, mean: float) -> np.ndarray:
    return np.full(shape=(n_rows,), fill_value=mean, dtype=float)


def main(input_csv: Optional[str] = None, output_csv: Optional[str] = None) -> None:
    mean = load_mean()
    if input_csv is None:
        # Create a 1-row dummy frame to show the value
        df = pd.DataFrame({"dummy": [1]})
    else:
        df = pd.read_csv(input_csv)

    preds = predict_const(len(df), mean)
    out = df.copy()
    out[f"{TARGET_NAME}_pred"] = preds

    if output_csv is None:
        output_path = MODEL_DIR / "manual_inference.csv"
    else:
        output_path = Path(output_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Loaded mean={mean:.4f}. Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
