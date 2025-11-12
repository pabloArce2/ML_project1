import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"


@dataclass
class TargetSetup:
    """Container describing which response variables are fitted in a configuration."""

    name: str
    targets: Sequence[str]
    description: str

    @property
    def is_multi_target(self) -> bool:
        return len(self.targets) > 1


@dataclass
class ExperimentResult:
    """Stores cross-validation outcomes for a single target configuration."""

    setup: TargetSetup
    lambdas: np.ndarray
    cv_errors: np.ndarray
    fold_errors: np.ndarray
    best_lambda: float
    best_error: float
    train_error: float
    test_error: float
    coefficients: pd.DataFrame
    intercept: pd.Series
    feature_names: List[str]
    model: Pipeline = field(repr=False)

    def to_dict(self) -> dict:
        """Lightweight serialization for downstream reporting."""
        return {
            "target": self.setup.name,
            "targets": list(self.setup.targets),
            "description": self.setup.description,
            "lambda_grid": self.lambdas.tolist(),
            "mean_cv_errors": self.cv_errors.tolist(),
            "fold_errors": self.fold_errors.tolist(),
            "best_lambda": self.best_lambda,
            "best_error": self.best_error,
            "train_error": self.train_error,
            "test_error": self.test_error,
            "coefficients": self.coefficients.to_dict(orient="index"),
            "intercept": self.intercept.to_dict(),
            "feature_names": self.feature_names,
        }


TARGET_SETUPS: List[TargetSetup] = [
    TargetSetup(
        name="ldl_only",
        targets=["ldl"],
        description="Predict LDL cholesterol using all remaining risk factors (SBP retained as a predictor).",
    ),
    TargetSetup(
        name="sbp_only",
        targets=["sbp"],
        description="Predict systolic blood pressure with LDL kept as an explanatory variable.",
    ),
    TargetSetup(
        name="joint_ldl_sbp",
        targets=["ldl", "sbp"],
        description="Jointly predict LDL and SBP; both variables are excluded from the feature matrix.",
    ),
]


def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    """Load the SAheart dataset from the supplied URL."""
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    df["famhist"] = df["famhist"].astype("category")
    return df


def make_design_matrix(df: pd.DataFrame, target_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the feature matrix for a specific prediction task.

    Parameters
    ----------
    df:
        Full SAheart dataframe.
    target_cols:
        Column names that will be predicted (removed from the feature matrix).
    """
    drop_cols = set(target_cols)
    drop_cols.add("chd")

    X_df = df.drop(columns=list(drop_cols), errors="ignore")
    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)

    y_df = df[list(target_cols)].copy()

    return X_df, y_df


def cross_val_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate generalization error for each lambda using K-fold CV."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_errors = np.zeros((len(lambdas), n_splits))

    for lambda_idx, alpha in enumerate(lambdas):
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("reg", Ridge(alpha=alpha, fit_intercept=True)),
                ]
            )
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            err = mean_squared_error(
                y[test_idx],
                preds,
                multioutput="uniform_average",
            )
            fold_errors[lambda_idx, fold_idx] = err

    mean_errors = fold_errors.mean(axis=1)
    return mean_errors, fold_errors


def fit_final_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    feature_names: Sequence[str],
    targets: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """
    Fit ridge regression on the full dataset and recover coefficients expressed in the original feature scale.

    Returns
    -------
    coeffs_df:
        DataFrame indexed by feature name with one column per target.
    intercept_series:
        Intercept in the original (unstandardized) feature space.
    model:
        Trained StandardScaler + Ridge pipeline ready for prediction.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    )
    model.fit(X, y)

    scaler: StandardScaler = model.named_steps["scaler"]
    reg: Ridge = model.named_steps["reg"]

    coef_standardized = np.atleast_2d(reg.coef_)
    scale = scaler.scale_.copy()
    scale_safe = np.where(scale == 0, 1.0, scale)
    coef_original = coef_standardized / scale_safe[np.newaxis, :]

    intercept_standardized = np.atleast_1d(reg.intercept_)
    intercept_original = intercept_standardized - coef_original @ scaler.mean_

    coeffs_df = pd.DataFrame(
        coef_original.T,
        index=feature_names,
        columns=list(targets),
    )
    intercept_series = pd.Series(intercept_original, index=list(targets))

    return coeffs_df, intercept_series, model


def describe_top_attributes(result: ExperimentResult, top_k: int = 5) -> str:
    """Generate a short textual interpretation of the strongest attributes."""
    lines = []
    coefs = result.coefficients
    for target in coefs.columns:
        sorted_coefs = coefs[target].abs().sort_values(ascending=False)
        top_features = sorted_coefs.index[:top_k]
        lines.append(f"Top attributes for {target} (|beta|):")
        for feat in top_features:
            direction = "increases" if coefs.loc[feat, target] > 0 else "decreases"
            lines.append(f"  - {feat}: {direction} predicted {target}")
    return "\n".join(lines)


def save_fold_errors(result_dir: Path, lambdas: np.ndarray, fold_errors: np.ndarray) -> Path:
    """Persist per-fold cross-validation errors for each lambda to CSV."""
    records = []
    for lambda_idx, alpha in enumerate(lambdas):
        for fold_idx, err in enumerate(fold_errors[lambda_idx]):
            records.append(
                {
                    "lambda": alpha,
                    "fold": fold_idx + 1,
                    "mse": err,
                }
            )
    df = pd.DataFrame(records)
    path = result_dir / "cv_errors.csv"
    df.to_csv(path, index=False)
    return path


def plot_cv_curve(result_dir: Path, setup: TargetSetup, lambdas: np.ndarray, mean_errors: np.ndarray, best_lambda: float, best_error: float) -> Path:
    """Plot the CV error as a function of lambda and save to disk."""
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, mean_errors, marker="o")
    plt.xscale("log")
    plt.xlabel("Regularization strength (lambda)")
    plt.ylabel("10-fold CV MSE")
    plt.title(f"{' & '.join(setup.targets)}: CV error vs lambda\nBest λ={best_lambda:.4g}, MSE={best_error:.4f}")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    path = result_dir / "lambda_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def run_single_setup(
    df: pd.DataFrame,
    setup: TargetSetup,
    lambdas: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
    save_predictions: bool = False,
    inference_df: Optional[pd.DataFrame] = None,
    inference_output: Optional[Path] = None,
) -> ExperimentResult:
    """Execute cross-validation (on the training split) and refit/testing for one target setting."""
    X_df, y_df = make_design_matrix(df, setup.targets)
    feature_names = X_df.columns.tolist()
    result_dir = RESULTS_DIR / setup.name
    result_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        X_df,
        y_df,
        test_size=0.1,
        random_state=random_state,
    )

    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)
    if setup.is_multi_target:
        y_train = y_train_df.values
        y_test = y_test_df.values
    else:
        y_train = y_train_df.iloc[:, 0].values
        y_test = y_test_df.iloc[:, 0].values

    mean_errors, fold_errors = cross_val_ridge(
        X_train, y_train, lambdas=lambdas, n_splits=n_splits, random_state=random_state
    )
    best_idx = int(np.argmin(mean_errors))
    best_lambda = float(lambdas[best_idx])
    best_error = float(mean_errors[best_idx])

    coeffs_df, intercept_series, model = fit_final_model(
        X=X_train,
        y=y_train,
        alpha=best_lambda,
        feature_names=feature_names,
        targets=setup.targets,
    )

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_error = float(
        mean_squared_error(y_train, train_preds, multioutput="uniform_average")
    )
    test_error = float(
        mean_squared_error(y_test, test_preds, multioutput="uniform_average")
    )

    result = ExperimentResult(
        setup=setup,
        lambdas=lambdas,
        cv_errors=mean_errors,
        fold_errors=fold_errors,
        best_lambda=best_lambda,
        best_error=best_error,
        train_error=train_error,
        test_error=test_error,
        coefficients=coeffs_df,
        intercept=intercept_series,
        feature_names=feature_names,
        model=model,
    )

    print(f"\n=== {setup.name} ===")
    print(setup.description)
    print(f"Average CV MSE (best λ={best_lambda:.4g}): {best_error:.4f}")
    print(f"Train MSE: {train_error:.4f} | Test MSE: {test_error:.4f}")
    print(describe_top_attributes(result))

    # Persist diagnostics
    errors_path = save_fold_errors(result_dir, lambdas, fold_errors)
    curve_path = plot_cv_curve(result_dir, setup, lambdas, mean_errors, best_lambda, best_error)
    print(f"Saved fold-by-fold errors to {errors_path}")
    print(f"Saved lambda curve to {curve_path}")

    # Persist model and feature names for downstream inference
    model_path = result_dir / "model.joblib"
    dump(model, model_path)
    feature_names_path = result_dir / "feature_names.json"
    with feature_names_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)
    print(f"Saved fitted model to {model_path}")

    targets_path = result_dir / "targets.json"
    with targets_path.open("w", encoding="utf-8") as handle:
        json.dump(list(setup.targets), handle, indent=2)

    if save_predictions:
        def preds_to_df(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
            array = values if values.ndim == 2 else values.reshape(-1, 1)
            return pd.DataFrame(array, index=index, columns=list(setup.targets))

        train_actual_df = y_train_df.copy()
        train_pred_df = preds_to_df(train_preds, train_actual_df.index)
        train_out = pd.concat(
            [train_actual_df.add_suffix("_actual"), train_pred_df.add_suffix("_pred")],
            axis=1,
        )
        train_path = result_dir / "train_predictions.csv"
        train_out.to_csv(train_path, index=True)
        print(f"Saved training predictions to {train_path}")

        test_actual_df = y_test_df.copy()
        test_pred_df = preds_to_df(test_preds, test_actual_df.index)
        test_out = pd.concat(
            [test_actual_df.add_suffix("_actual"), test_pred_df.add_suffix("_pred")],
            axis=1,
        )
        test_path = result_dir / "test_predictions.csv"
        test_out.to_csv(test_path, index=True)
        print(f"Saved test predictions to {test_path}")

    if inference_df is not None:
        inf_features = inference_df.copy()
        for col in feature_names:
            if col not in inf_features.columns:
                inf_features[col] = 0.0
        inf_features = inf_features[feature_names]

        preds = model.predict(inf_features.values)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]
        preds_df = pd.DataFrame(preds, columns=setup.targets)
        output_df = pd.concat([inference_df.reset_index(drop=True), preds_df], axis=1)

        if inference_output is None:
            inference_path = result_dir / "custom_inference.csv"
        else:
            inference_path = inference_output
            if inference_path.exists() and inference_path.is_dir():
                inference_path = inference_path / f"{setup.name}_inference_predictions.csv"

        output_df.to_csv(inference_path, index=False)
        print(f"Saved inference predictions to {inference_path}")

    return result


def run_experiments(
    selected_setups: Sequence[str],
    lambdas: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
    save_predictions: bool = False,
    inference_df: Optional[pd.DataFrame] = None,
    inference_output: Optional[Path] = None,
) -> List[ExperimentResult]:
    """Run ridge regression experiments for the requested target settings."""
    setup_lookup = {setup.name: setup for setup in TARGET_SETUPS}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "all" in selected_setups:
        chosen = TARGET_SETUPS
    else:
        missing = [name for name in selected_setups if name not in setup_lookup]
        if missing:
            raise ValueError(f"Unknown settings requested: {', '.join(missing)}")
        chosen = [setup_lookup[name] for name in selected_setups]

    df = load_saheart()
    results: List[ExperimentResult] = []
    for setup in chosen:
        result = run_single_setup(
            df,
            setup,
            lambdas=lambdas,
            n_splits=n_splits,
            random_state=random_state,
            save_predictions=save_predictions,
            inference_df=inference_df,
            inference_output=inference_output,
        )
        results.append(result)

    summary_path = RESULTS_DIR / "ridge_results.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump([res.to_dict() for res in results], handle, indent=2)
    print(f"\nSaved numeric summary to {summary_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge regression study for LDL/SBP predictions.")
    parser.add_argument(
        "--setting",
        choices=["all"] + [setup.name for setup in TARGET_SETUPS],
        nargs="+",
        default=["all"],
        help="Pick which regression settings to run (default: all). Use multiple values to combine runs.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10).",
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=1e-4,
        dest="lambda_min",
        help="Lower bound of the logarithmic lambda grid (default: 1e-4).",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=1e3,
        dest="lambda_max",
        help="Upper bound of the logarithmic lambda grid (default: 1e3).",
    )
    parser.add_argument(
        "--lambda-count",
        type=int,
        default=30,
        dest="lambda_count",
        help="Number of lambda values sampled between the bounds (default: 30).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist fitted predictions (actual vs. predicted) for each setting to CSV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed controlling train/test splitting and cross-validation shuffling.",
    )
    parser.add_argument(
        "--inference-input",
        type=str,
        default=None,
        help="Path to a CSV file containing feature columns to score with the fitted model.",
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
    lambdas = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), num=args.lambda_count)
    inference_df: Optional[pd.DataFrame] = None
    if args.inference_input:
        inference_df = pd.read_csv(args.inference_input)
    inference_output = Path(args.inference_output) if args.inference_output else None
    run_experiments(
        selected_setups=args.setting,
        lambdas=lambdas,
        n_splits=args.folds,
        random_state=args.random_state,
        save_predictions=args.save_predictions,
        inference_df=inference_df,
        inference_output=inference_output,
    )


if __name__ == "__main__":
    main()
