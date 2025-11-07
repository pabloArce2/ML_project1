
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load dataset from URL
DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"
TARGET_NAME = "chd"  # binary classification target (0/1)

@dataclass
class CVResult:
    fold_acc: np.ndarray
    mean_acc: float
    std_acc: float
    majority_class: int
    majority_frac: float

    def to_dict(self) -> dict:
        return {
            "model": "majority_class_baseline",
            "description": "Baseline that always predicts the most frequent class in y_train.",
            "target": TARGET_NAME,
            "fold_acc": self.fold_acc.tolist(),
            "mean_acc": float(self.mean_acc),
            "std_acc": float(self.std_acc),
            "majority_class": int(self.majority_class),
            "majority_frac": float(self.majority_frac),
        }

class MajorityClassBaseline:
    """Always predicts the majority class seen during fit()."""
    def __init__(self) -> None:
        self.class_: int | None = None
        self.frac_: float | None = None

    def fit(self, y: np.ndarray) -> "MajorityClassBaseline":
        # y may be {0,1} or strings; ensure integers 0/1
        y = np.asarray(y)
        # handle ties deterministically: pick the smaller class label on tie
        values, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        top = values[counts == max_count]
        self.class_ = int(np.min(top))
        self.frac_ = float(max_count / y.size)
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.class_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(shape=(n,), fill_value=self.class_, dtype=int)

def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    # index column is the row id
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    return df

def make_target(df: pd.DataFrame) -> np.ndarray:
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target '{TARGET_NAME}' not found. Columns: {list(df.columns)}")
    # Ensure binary 0/1 integers
    y = df[TARGET_NAME].to_numpy()
    # dataset has chd already as 0/1; cast to int just in case
    return y.astype(int, copy=False)

def plot_fold_accuracy(
    fold_acc: np.ndarray,
    output_path: Path | str = RESULTS_DIR / "majority_baseline_classification" / "baseline_classification_fold_acc.png",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds = np.arange(1, len(fold_acc) + 1)
    plt.plot(folds, fold_acc, marker="o", linestyle="-")
    plt.title("Baseline Classification: Accuracy across folds")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.savefig(output_path, dpi=300)
    plt.close()

def cross_validate_majority(y: np.ndarray, n_splits: int = 10, random_state: int = 42) -> Tuple[np.ndarray, float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_acc = np.zeros(n_splits, dtype=float)
    for i, (train_idx, test_idx) in enumerate(kf.split(y)):
        y_train, y_test = y[train_idx], y[test_idx]
        model = MajorityClassBaseline().fit(y_train)
        preds = model.predict(len(test_idx))
        fold_acc[i] = accuracy_score(y_test, preds)
    return fold_acc, float(fold_acc.mean()), float(fold_acc.std(ddof=1))

def save_artifacts(result_dir: Path, result: CVResult) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    # Minimal "model" artefact: store majority class + fraction
    model_path = result_dir / "model.joblib"
    dump({"type": "majority_baseline", "target": TARGET_NAME,
          "majority_class": result.majority_class, "majority_frac": result.majority_frac}, model_path)
    # Targets file to mirror other modules
    targets_path = result_dir / "targets.json"
    targets_path.write_text(json.dumps([TARGET_NAME], indent=2))
    # Summary JSON
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2))

def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_saheart()
    y = make_target(df)

    # CV evaluation
    fold_acc, mean_acc, std_acc = cross_validate_majority(
        y, n_splits=args.folds, random_state=args.random_state
    )

    # Fit on full data to record the global majority (for inference)
    model = MajorityClassBaseline().fit(y)
    majority_class = model.class_
    majority_frac = model.frac_

    # Plot & save CV metrics
    plot_fold_accuracy(fold_acc)
    result_dir = RESULTS_DIR / "majority_baseline_classification"
    cv_path = result_dir / "cv_metrics.csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fold": np.arange(1, len(fold_acc) + 1), "accuracy": fold_acc}).to_csv(cv_path, index=False)

    # Optionally save in-sample predictions (sanity check)
    if args.save_predictions:
        preds = model.predict(len(y))
        pred_path = result_dir / "predictions.csv"
        pd.DataFrame({"actual": y, "predicted": preds}).to_csv(pred_path, index=True)

    # Persist artefacts + summary (and a top-level summary for parity)
    res = CVResult(fold_acc=fold_acc, mean_acc=mean_acc, std_acc=std_acc,
                   majority_class=majority_class, majority_frac=majority_frac)
    save_artifacts(result_dir, res)
    summary_path = RESULTS_DIR / "classification_results.json"
    summary_path.write_text(json.dumps([res.to_dict()], indent=2))

    # Console summary
    print("\n=== Majority-Class Baseline (Classification) ===")
    print(f"Global majority class for '{TARGET_NAME}': {majority_class} (fraction={majority_frac:.3f})")
    print(f"{args.folds}-Fold CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Saved CV metrics to {cv_path}")
    if args.save_predictions:
        print(f"Saved fitted predictions to {pred_path}")
    print(f"Saved model artefacts and summary to {result_dir}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline classification using majority class (evaluate with accuracy).")
    p.add_argument("--folds", type=int, default=10, help="Number of CV folds (default: 10).")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for shuffling (default: 42).")
    p.add_argument("--save-predictions", action="store_true", help="Persist in-sample predictions as CSV.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()
