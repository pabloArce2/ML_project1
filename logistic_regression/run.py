import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"
TARGET_NAME = "chd"


@dataclass
class ModelSetup:
    """Description of a classification model and the hyper-parameter grid to explore."""

    name: str
    description: str
    param_name: str
    param_values: Sequence[float]
    build_estimator: Callable[[float], Pipeline]


@dataclass
class ModelResult:
    """Stores cross-validation diagnostics and the refitted model for a setup."""

    setup: ModelSetup
    mean_accuracy: np.ndarray
    mean_log_loss: np.ndarray
    fold_metrics: np.ndarray
    best_param: float
    best_accuracy: float
    best_log_loss: float
    train_accuracy: float
    train_log_loss: float
    test_accuracy: float
    test_log_loss: float
    model: Pipeline
    feature_names: List[str]

    def to_dict(self) -> dict:
        return {
            "model": self.setup.name,
            "description": self.setup.description,
            "param_name": self.setup.param_name,
            "param_grid": list(self.setup.param_values),
            "mean_accuracy": self.mean_accuracy.tolist(),
            "mean_log_loss": self.mean_log_loss.tolist(),
            "best_param": self.best_param,
            "best_accuracy": self.best_accuracy,
            "best_log_loss": self.best_log_loss,
            "train_accuracy": self.train_accuracy,
            "train_log_loss": self.train_log_loss,
            "test_accuracy": self.test_accuracy,
            "test_log_loss": self.test_log_loss,
            "feature_names": self.feature_names,
        }


def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    df["famhist"] = df["famhist"].astype("category")
    return df


def make_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X_df = df.drop(columns=[TARGET_NAME])
    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)
    y = df[TARGET_NAME].to_numpy(dtype=int)
    return X_df, y


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    setup: ModelSetup,
    n_splits: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = np.zeros((len(setup.param_values), n_splits, 2))

    for param_idx, value in enumerate(setup.param_values):
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            estimator = setup.build_estimator(value)
            estimator.fit(X[train_idx], y[train_idx])
            preds = estimator.predict(X[test_idx])
            proba = estimator.predict_proba(X[test_idx])[:, 1]
            proba = np.clip(proba, 1e-9, 1 - 1e-9)
            acc = accuracy_score(y[test_idx], preds)
            loss = log_loss(y[test_idx], proba, labels=[0, 1])
            fold_metrics[param_idx, fold_idx, 0] = acc
            fold_metrics[param_idx, fold_idx, 1] = loss

    mean_accuracy = fold_metrics[:, :, 0].mean(axis=1)
    mean_log_loss = fold_metrics[:, :, 1].mean(axis=1)
    return mean_accuracy, mean_log_loss, fold_metrics


def fit_final_model(
    X: np.ndarray,
    y: np.ndarray,
    setup: ModelSetup,
    param: float,
) -> Pipeline:
    model = setup.build_estimator(param)
    model.fit(X, y)
    return model


def save_fold_metrics(
    result_dir: Path,
    setup: ModelSetup,
    fold_metrics: np.ndarray,
) -> Path:
    records = []
    for param_idx, value in enumerate(setup.param_values):
        for fold_idx, (acc, loss) in enumerate(fold_metrics[param_idx]):
            records.append(
                {
                    setup.param_name: value,
                    "fold": fold_idx + 1,
                    "accuracy": float(acc),
                    "log_loss": float(loss),
                }
            )
    df = pd.DataFrame.from_records(records)
    path = result_dir / "cv_metrics.csv"
    df.to_csv(path, index=False)
    return path


def plot_accuracy_curve(
    result_dir: Path,
    setup: ModelSetup,
    mean_accuracy: np.ndarray,
) -> Path:
    plt.figure(figsize=(7, 5))
    plt.plot(setup.param_values, mean_accuracy, marker="o")
    if setup.param_name == "lambda":
        plt.xscale("log")
        x_label = "Regularisation strength (λ)"
    else:
        x_label = setup.param_name
    plt.xlabel(x_label)
    plt.ylabel("10-fold CV accuracy")
    plt.title(f"{setup.name}: CV accuracy vs {setup.param_name}")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    path = result_dir / "accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_model_artifacts(
    result_dir: Path,
    result: ModelResult,
) -> None:
    model_path = result_dir / "model.joblib"
    dump(result.model, model_path)
    feature_names_path = result_dir / "feature_names.json"
    with feature_names_path.open("w", encoding="utf-8") as handle:
        json.dump(result.feature_names, handle, indent=2)
    targets_path = result_dir / "targets.json"
    with targets_path.open("w", encoding="utf-8") as handle:
        json.dump([TARGET_NAME], handle, indent=2)
    summary_path = result_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    print(f"Saved model artefacts to {result_dir}")


def run_single_model(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: List[str],
    setup: ModelSetup,
    n_splits: int,
    random_state: int,
    save_predictions: bool,
    inference_df: pd.DataFrame | None,
    inference_output: Path | None,
) -> ModelResult:
    result_dir = RESULTS_DIR / setup.name
    result_dir.mkdir(parents=True, exist_ok=True)

    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)
    y_train_arr = y_train
    y_test_arr = y_test

    mean_acc, mean_loss, fold_metrics = cross_validate_model(
        X_train, y_train_arr, setup, n_splits=n_splits, random_state=random_state
    )
    best_idx = int(np.argmax(mean_acc))
    best_param = float(setup.param_values[best_idx])
    best_accuracy = float(mean_acc[best_idx])
    best_log_loss = float(mean_loss[best_idx])

    model = fit_final_model(X_train, y_train_arr, setup, best_param)

    train_preds = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    train_proba = np.clip(train_proba, 1e-9, 1 - 1e-9)
    train_accuracy = float(accuracy_score(y_train_arr, train_preds))
    train_log_loss = float(log_loss(y_train_arr, train_proba, labels=[0, 1]))

    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_proba = np.clip(test_proba, 1e-9, 1 - 1e-9)
    test_accuracy = float(accuracy_score(y_test_arr, test_preds))
    test_log_loss = float(log_loss(y_test_arr, test_proba, labels=[0, 1]))

    result = ModelResult(
        setup=setup,
        mean_accuracy=mean_acc,
        mean_log_loss=mean_loss,
        fold_metrics=fold_metrics,
        best_param=best_param,
        best_accuracy=best_accuracy,
        best_log_loss=best_log_loss,
        train_accuracy=train_accuracy,
        train_log_loss=train_log_loss,
        test_accuracy=test_accuracy,
        test_log_loss=test_log_loss,
        model=model,
        feature_names=feature_names,
    )

    print(f"\n=== {setup.name} ===")
    print(setup.description)
    print(
        f"Best {setup.param_name}={best_param} with mean CV accuracy {best_accuracy:.4f} "
        f"(log-loss {best_log_loss:.4f})"
    )
    print(f"Train accuracy: {train_accuracy:.4f} | Test accuracy: {test_accuracy:.4f}")
    print(f"Train log-loss: {train_log_loss:.4f} | Test log-loss: {test_log_loss:.4f}")

    metrics_path = save_fold_metrics(result_dir, setup, fold_metrics)
    curve_path = plot_accuracy_curve(result_dir, setup, mean_acc)
    print(f"Saved fold metrics to {metrics_path}")
    print(f"Saved CV accuracy curve to {curve_path}")

    if save_predictions:
        train_df = pd.DataFrame(
            {
                "actual": y_train_arr,
                "predicted": train_preds,
                "proba_chd": train_proba,
            },
            index=X_train_df.index,
        )
        train_path = result_dir / "train_predictions.csv"
        train_df.to_csv(train_path, index=True)
        print(f"Saved training predictions to {train_path}")

        test_df = pd.DataFrame(
            {
                "actual": y_test_arr,
                "predicted": test_preds,
                "proba_chd": test_proba,
            },
            index=X_test_df.index,
        )
        test_path = result_dir / "test_predictions.csv"
        test_df.to_csv(test_path, index=True)
        print(f"Saved test predictions to {test_path}")

    if inference_df is not None:
        aligned = inference_df.copy()
        for col in feature_names:
            if col not in aligned.columns:
                aligned[col] = 0.0
        aligned = aligned[feature_names]
        inf_preds = model.predict_proba(aligned.values)[:, 1]
        inf_preds = np.clip(inf_preds, 1e-9, 1 - 1e-9)
        inf_labels = model.predict(aligned.values)
        output_df = inference_df.reset_index(drop=True).assign(
            chd_probability=inf_preds,
            chd_prediction=inf_labels,
        )
        if inference_output is None:
            inference_path = result_dir / "custom_inference.csv"
        else:
            inference_path = inference_output
            if inference_path.exists() and inference_path.is_dir():
                inference_path = inference_path / f"{setup.name}_inference.csv"
        output_df.to_csv(inference_path, index=False)
        print(f"Saved inference predictions to {inference_path}")

    save_model_artifacts(result_dir, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification study predicting CHD with penalised logistic regression.")
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of stratified CV folds (default: 10).",
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=1e-5,
        dest="lambda_min",
        help="Lower bound for the λ grid (default: 1e-5).",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=1e4,
        dest="lambda_max",
        help="Upper bound for the λ grid (default: 1e4).",
    )
    parser.add_argument(
        "--lambda-count",
        type=int,
        default=60,
        dest="lambda_count",
        help="Number of λ values sampled logarithmically (default: 60).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed controlling cross-validation shuffling (default: 42).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist fitted predictions for the full dataset.",
    )
    parser.add_argument(
        "--inference-input",
        type=str,
        default=None,
        help="CSV with feature columns to score.",
    )
    parser.add_argument(
        "--inference-output",
        type=str,
        default=None,
        help="Optional output path (file or directory) for inference predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambda_grid = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), num=args.lambda_count)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_saheart()
    X_df, y = make_design_matrix(df)
    feature_names = X_df.columns.tolist()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.1,
        stratify=y,
        random_state=args.random_state,
    )

    inference_df = None
    if args.inference_input:
        inference_df = pd.read_csv(args.inference_input)
    inference_output = Path(args.inference_output) if args.inference_output else None

    logistic_setup = ModelSetup(
        name="logistic_regression",
        description="Standardised features with L2-regularised logistic regression sweeping λ.",
        param_name="lambda",
        param_values=list(lambda_grid),
        build_estimator=lambda lam: Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0 / lam,
                        solver="liblinear",
                        max_iter=1000,
                        class_weight=None,
                    ),
                ),
            ]
        ),
    )

    result = run_single_model(
        X_train_df=X_train_df,
        y_train=y_train,
        X_test_df=X_test_df,
        y_test=y_test,
        feature_names=feature_names,
        setup=logistic_setup,
        n_splits=args.folds,
        random_state=args.random_state,
        save_predictions=args.save_predictions,
        inference_df=inference_df,
        inference_output=inference_output,
    )

    summary_path = RESULTS_DIR / "classification_results.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump([result.to_dict()], handle, indent=2)
    print(f"\nSaved summary of all runs to {summary_path}")


if __name__ == "__main__":
    main()
