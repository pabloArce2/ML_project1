"""
Utility script to emit baseline predictions for any provided rows.
Since this baseline ignores features, it only loads the stored mean and
produces a constant column "<target>_pred".
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load

from .run import RESULTS_DIR


def get_model_dir(target: str) -> Path:
    """
    Locate the directory that contains the mean-baseline artefacts for a given target.
    Example: target="ldl" -> results/mean_baseline_regression_ldl
             target="sbp" -> results/mean_baseline_regression_sbp
    """
    return RESULTS_DIR / f"mean_baseline_regression_{target}"


def load_mean(target: str) -> float:
    model_dir = get_model_dir(target)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing baseline artefacts for target '{target}'. "
            f"Run: python -m baseline_regression.run --target {target}"
        )
    data = load(model_path)
    mean = float(data["mean"])
    return mean


def predict_const(n_rows: int, mean: float) -> np.ndarray:
    return np.full(shape=(n_rows,), fill_value=mean, dtype=float)


def main(
    input_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    target: str = "ldl",
) -> None:
    mean = load_mean(target)
    if input_csv is None:
        # Create a 1-row dummy frame to show the value
        df = pd.DataFrame({"dummy": [1]})
    else:
        df = pd.read_csv(input_csv)

    preds = predict_const(len(df), mean)
    out = df.copy()
    out[f"{target}_pred"] = preds

    if output_csv is None:
        model_dir = get_model_dir(target)
        output_path = model_dir / "manual_inference.csv"
    else:
        output_path = Path(output_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"[{target}] Loaded mean={mean:.4f}. Wrote predictions to {output_path}")


def _cli():
    parser = argparse.ArgumentParser(
        description="Emit constant baseline predictions for a target using the stored mean."
    )
    parser.add_argument(
        "--target",
        choices=["ldl", "sbp"],
        default="ldl",
        help="Target variable whose baseline you want to use.",
    )
    parser.add_argument(
        "--input",
        dest="input_csv",
        default=None,
        help="Optional input CSV. If omitted, a 1-row dummy file is used.",
    )
    parser.add_argument(
        "--output",
        dest="output_csv",
        default=None,
        help="Output CSV path. If omitted, writes manual_inference.csv under the model dir.",
    )
    args = parser.parse_args()
    main(input_csv=args.input_csv, output_csv=args.output_csv, target=args.target)


if __name__ == "__main__":
    _cli()
