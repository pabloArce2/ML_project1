import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Dataset + paths (reuse the same SAheart dataset used in the classification examples)
DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"
TARGET_NAME = "ldl"


@dataclass
class CVResult:
    fold_mse: np.ndarray
    mean_mse: float
    std_mse: float
    global_mean: float

    def to_dict(self) -> dict:
        return {
            "model": "mean_baseline",
            "description": "Baseline that always predicts the training-mean of the target.",
            "target": TARGET_NAME,
            "fold_mse": self.fold_mse.tolist(),
            "mean_mse": float(self.mean_mse),
            "std_mse": float(self.std_mse),
            "global_mean": float(self.global_mean),
        }


class MeanBaseline:
    """A tiny regressor that predicts the mean of y seen during fit()."""

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, y: np.ndarray) -> "MeanBaseline":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(shape=(n,), fill_value=self.mean_, dtype=float)


def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    # index is an id column
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    # No transformations needed for a mean baseline
    return df


def make_target(df: pd.DataFrame) -> np.ndarray:
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target '{TARGET_NAME}' not found in dataset columns: {list(df.columns)}")
    return df[TARGET_NAME].to_numpy(dtype=float)

def plot_fold_mse(
    fold_mse,
    output_path: Path | str = RESULTS_DIR / "mean_baseline_regression" / "baseline_regression_fold_mse.png",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds = np.arange(1, len(fold_mse) + 1)
    plt.plot(folds, fold_mse, marker='o', linestyle='-', color='b')
    plt.title('Baseline Regression: MSE across folds')
    plt.xlabel('Fold Number')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_path, dpi=300)
    plt.close() 

def cross_validate_mean_baseline(y: np.ndarray, n_splits: int = 10, random_state: int = 42) -> Tuple[np.ndarray, float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_mse = np.zeros(n_splits, dtype=float)

    for i, (train_idx, test_idx) in enumerate(kf.split(y)):
        y_train, y_test = y[train_idx], y[test_idx]
        model = MeanBaseline().fit(y_train)
        preds = model.predict(len(test_idx))
        fold_mse[i] = mean_squared_error(y_test, preds)

    return fold_mse, float(fold_mse.mean()), float(fold_mse.std(ddof=1))


def save_artifacts(result_dir: Path, result: CVResult) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save a tiny "model" that only contains the global mean
    model_path = result_dir / "model.joblib"
    dump({"type": "mean_baseline", "target": TARGET_NAME, "mean": result.global_mean}, model_path)

    # Save targets + summary for parity with the classification package
    targets_path = result_dir / "targets.json"
    with targets_path.open("w", encoding="utf-8") as fh:
        json.dump([TARGET_NAME], fh, indent=2)

    summary_path = result_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_saheart()
    y = make_target(df)

    # CV evaluation
    fold_mse, mean_mse, std_mse = cross_validate_mean_baseline(
        y, n_splits=args.folds, random_state=args.random_state
    )

    model = MeanBaseline().fit(y)
    global_mean = model.mean_

    plot_fold_mse(fold_mse)


    # Save CV metrics
    result_dir = RESULTS_DIR / "mean_baseline_regression"
    cv_path = result_dir / "cv_metrics.csv"

    result_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"fold": np.arange(1, len(fold_mse) + 1), "mse": fold_mse}).to_csv(cv_path, index=False)

    # Optionally save in-sample predictions (useful to sanity check)
    if args.save_predictions:
        preds = model.predict(len(y))
        pred_path = result_dir / "predictions.csv"
        pd.DataFrame({"actual": y, "predicted": preds}).to_csv(pred_path, index=True)

    # Persist artefacts + summary
    res = CVResult(fold_mse=fold_mse, mean_mse=mean_mse, std_mse=std_mse, global_mean=global_mean)
    save_artifacts(result_dir, res)

    # Also save a top-level summary like in the classification package
    summary_path = RESULTS_DIR / "regression_results.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump([res.to_dict()], fh, indent=2)

    print("\n=== Mean Baseline (Regression) ===")
    print(f"Global mean of '{TARGET_NAME}': {global_mean:.4f}")
    print(f"{args.folds}-Fold CV MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"Saved CV metrics to {cv_path}")
    if args.save_predictions:
        print(f"Saved fitted predictions to {pred_path}")
    print(f"Saved model artefacts and summary to {result_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline regression using the training-mean (evaluate with MSE).")
    p.add_argument("--folds", type=int, default=10, help="Number of CV folds (default: 10).")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for shuffling (default: 42).")
    p.add_argument("--save-predictions", action="store_true", help="Persist in-sample predictions as CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
