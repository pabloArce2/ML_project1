# ann_regression/run.py  — ANN Part (Regression, part b)
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from joblib import dump
import argparse 


DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"


@dataclass
class TargetSetup:
    name: str
    targets: Sequence[str]
    description: str

    @property
    def is_multi_target(self) -> bool:
        return len(self.targets) > 1

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
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    df["famhist"] = df["famhist"].astype("category")
    return df

def make_design_matrix(df: pd.DataFrame, target_cols: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    drop_cols = set(target_cols)
    drop_cols.add("chd")  
    X_df = df.drop(columns=list(drop_cols), errors="ignore")
    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)

    y_df = df[list(target_cols)]
    y = y_df.values if len(target_cols) > 1 else y_df.values.ravel()
    return X_df, y

def build_ann_pipeline(h: int, alpha: float, random_state: int = 0) -> Pipeline:
    """
    Standardize features, then 1-hidden-layer MLP for regression.
    - h: hidden units (complexity parameter)
    - alpha: L2 regularization strength (λ)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(h,),
            alpha=alpha,
            max_iter=10000,
            early_stopping=True,
            n_iter_no_change=20,
            tol=1e-4,
            random_state=random_state,
        )),
    ])


def inner_cv_mse(X: pd.DataFrame, y: np.ndarray, h: int, alpha: float,
                 n_splits: int = 5, random_state: int = 0) -> float:
    """
    Mean CV MSE for a single (h, alpha) using KFold.
    """
    model = build_ann_pipeline(h=h, alpha=alpha, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    return float(-scores.mean())


def nested_cv_ann(
    X: pd.DataFrame,
    y: np.ndarray,
    h_grid: Sequence[int],
    alpha_grid: Sequence[float],
    n_outer: int = 10,
    n_inner: int = 5,
    random_state: int = 0,
):
    """
    Two-level CV for ANN:
      - Outer loop: split data into train/test.
      - Inner loop: choose (h*, alpha*) by inner-CV on the outer-train split.
    Saves:
      - performance_by_h_alpha.csv : all inner-CV results (for plotting)
    Returns:
      - List of dicts with outer_fold, h_star, alpha_star, Etest_ann, Etest_baseline
    """
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    results = []        # stores final outer test performance
    inner_records = []  # stores all inner-CV combinations for plotting

    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = (y[tr_idx], y[te_idx]) if y.ndim == 1 else (y[tr_idx], y[te_idx])

        # --- inner loop: select (h*, alpha*) ---
        best_h, best_a, best_cv = None, None, np.inf
        for h in h_grid:
            for a in alpha_grid:
                cv = inner_cv_mse(X_tr, y_tr, h=h, alpha=a, n_splits=n_inner, random_state=random_state)

                # record all tested pairs for plotting
                inner_records.append({
                    "outer_fold": fold_idx,
                    "h": h,
                    "alpha": a,
                    "inner_cv_mse": cv,
                })

                if cv < best_cv:
                    best_cv, best_h, best_a = cv, h, a

        # --- train final model on outer-train with (h*, a*) and evaluate ---
        model = build_ann_pipeline(h=best_h, alpha=best_a, random_state=random_state)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
            y_te_eval = y_te.ravel()
        else:
            y_te_eval = y_te

        Etest_ann = float(mean_squared_error(y_te_eval, y_pred))

        # --- baseline on same outer split ---
        y_mean = float(np.mean(y_tr if y_tr.ndim == 1 else y_tr.ravel()))
        Etest_baseline = float(np.mean((y_te_eval - y_mean) ** 2))

        # store final outer results
        results.append({
            "outer_fold": fold_idx,
            "h_star": best_h,
            "alpha_star": best_a,
            "Etest_ann": Etest_ann,
            "Etest_baseline": Etest_baseline,
        })

        print(f"[fold {fold_idx}] best_h={best_h}, best_alpha={best_a}, testMSE={Etest_ann:.4f}")

    # --- save inner-CV results for plotting ---
    pd.DataFrame(inner_records).to_csv(RESULTS_DIR / "performance_by_h_alpha.csv", index=False)

    return results


def fit_and_save_final_ann(
    X_df: pd.DataFrame,
    y: np.ndarray,
    setup: TargetSetup,
    h_grid: Sequence[int],
    alpha_grid: Sequence[float],
    n_splits: int = 10,
    results_dir: Path = RESULTS_DIR,
    save_predictions: bool = False,
    inference_df: Optional[pd.DataFrame] = None,
    inference_output: Optional[Path] = None,
    random_state: int = 0,
) -> dict:
    """
    Select (h*, alpha*) via CV on the full dataset, fit final ANN, and persist artefacts:
    - model.joblib, feature_names.json, targets.json
    - optional predictions.csv on training data
    - optional custom inference CSV
    """
    X = X_df.values
    feature_names = list(X_df.columns)

    # simple CV selection on the full data (not nested) to get a single (h*, alpha*)
    best_h, best_a, best_cv = None, None, np.inf
    for h in h_grid:
        for a in alpha_grid:
            cv = inner_cv_mse(X_df, y, h=h, alpha=a, n_splits=n_splits, random_state=random_state)
            if cv < best_cv:
                best_cv, best_h, best_a = cv, h, a

    # fit final model on all data
    model = build_ann_pipeline(h=best_h, alpha=best_a, random_state=random_state)
    model.fit(X, y)

    # persist artefacts
    result_dir = results_dir / setup.name
    result_dir.mkdir(parents=True, exist_ok=True)

    model_path = result_dir / "model.joblib"
    dump(model, model_path)

    feature_names_path = result_dir / "feature_names.json"
    with feature_names_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    targets_path = result_dir / "targets.json"
    with targets_path.open("w", encoding="utf-8") as handle:
        json.dump(list(setup.targets), handle, indent=2)

    # save train predictions
    if save_predictions:
        preds = model.predict(X)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]
        preds_df = pd.DataFrame(preds, index=X_df.index, columns=setup.targets)
        actual_df = pd.DataFrame(y if y.ndim > 1 else y[:, np.newaxis], index=X_df.index, columns=setup.targets)
        out_df = pd.concat([actual_df.add_suffix("_actual"), preds_df.add_suffix("_pred")], axis=1)
        out_df.to_csv(result_dir / "predictions.csv", index=True)

    #run custom inference from a CSV of raw features (columns need to match/align)
    if inference_df is not None:
        inf_features = inference_df.copy()
        # align columns to training features (fill missing with 0)
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

    return {"h_star": best_h, "alpha_star": best_a, "cv_mse": best_cv, "model_path": str(model_path)}


def main():

    parser = argparse.ArgumentParser(description="ANN regression — nested CV + optional final fit")
    parser.add_argument("--setting", default="ldl_only", help="Target setup name (ldl_only | sbp_only | joint_ldl_sbp)")
    parser.add_argument("--outer", type=int, default=5, help="Outer folds K1 (spec says 10; use 5 for speed during dev)")
    parser.add_argument("--inner", type=int, default=5, help="Inner folds K2 (spec says 10; use 5 for speed during dev)")
    parser.add_argument("--h-grid", nargs="+", type=int, default=[1, 4, 8], help="Hidden units to try")
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.001, 0.01, 0.1], help="Alphas (λ) to try")
    parser.add_argument("--fit-final", action="store_true", help="Fit final ANN on full data using CV-selected (h*, α*) and save artefacts")
    parser.add_argument("--save-predictions", action="store_true", help="When fitting final model, also save train predictions.csv")
    args = parser.parse_args()

    # 1) load & prepare
    df = load_saheart()
    setup = next(s for s in TARGET_SETUPS if s.name == args.setting)
    X_df, y = make_design_matrix(df, setup.targets)

    # 2) nested CV summary table  
    rows = nested_cv_ann(
        X=X_df, y=y,
        h_grid=args.h_grid, alpha_grid=args.alpha_grid,
        n_outer=args.outer, n_inner=args.inner, random_state=0
    )
    out = pd.DataFrame(rows, columns=["outer_fold","h_star","alpha_star","Etest_ann","Etest_baseline"])
    print("\nTwo-level CV (ANN vs baseline) —", args.setting)
    print(out.to_string(index=False))
    print(f"\nMean outer test MSE (ANN): {out['Etest_ann'].mean():.4f}")
    print(f"Mean outer test MSE (baseline): {out['Etest_baseline'].mean():.4f}")

    out.to_csv(RESULTS_DIR / f"{setup.name}_nested_cv_ann.csv", index=False)
 

    # 3) fit + save final model/artefacts for inference (model.joblib, feature_names.json, targets.json)
    if args.fit_final:
        summary = fit_and_save_final_ann(
            X_df=X_df, y=y, setup=setup,
            h_grid=args.h_grid, alpha_grid=args.alpha_grid,
            n_splits=args.inner,  # reuse inner folds for final selection
            results_dir=RESULTS_DIR,
            save_predictions=args.save_predictions,
            inference_df=None, inference_output=None,
            random_state=0,
        )
        print(f"\n[ok] Final model saved — h*={summary['h_star']}, α*={summary['alpha_star']}, cv_mse={summary['cv_mse']:.4f}")
        print(f"Model path: {summary['model_path']}")

        # --- create and save performance vs α plot ---
    perf_path = RESULTS_DIR / "performance_by_h_alpha.csv"
    if perf_path.exists():
        df = pd.read_csv(perf_path)
        avg = df.groupby(["h", "alpha"], as_index=False)["inner_cv_mse"].mean()

        plt.figure(figsize=(8,6))
        for h, group in avg.groupby("h"):
            plt.plot(group["alpha"], group["inner_cv_mse"], marker="o", label=f"h={h}")

        plt.xscale("log")
        plt.xlabel("Regularization α (log scale)")
        plt.ylabel("Mean Inner-CV MSE")
        plt.title("ANN performance vs regularization (per hidden layer size)")
        plt.legend()
        plt.tight_layout()

        out_path = RESULTS_DIR / "performance_by_h_alpha.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[ok] Saved performance plot → {out_path}")

if __name__ == "__main__":
    main()






