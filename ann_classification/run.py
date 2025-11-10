# ann_classification/run.py — Classification part (baseline vs logistic vs ANN)
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from math import fabs
from scipy.stats import chi2  # for McNemar p-values

DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"
SETUP_NAME = "chd_only"   # binary classification target


def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    df["famhist"] = df["famhist"].astype("category")
    # Ensure binary target
    assert set(df["chd"].unique()) <= {0, 1}, "Expected chd to be binary 0/1."
    return df


def make_design_matrix_for_chd(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    # Predict chd from all other variables
    X_df = df.drop(columns=["chd"], errors="ignore")
    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)
    y = df["chd"].values.astype(int)
    return X_df, y


# -------------------- Pipelines --------------------

def build_logreg_pipeline(lmbda: float, random_state: int = 0) -> Pipeline:
    # In sklearn: C = 1/λ ; handle λ=0 (no reg) by using a very large C
    C = 1e12 if lmbda == 0 else 1.0 / lmbda
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=10000,
            random_state=random_state,
        )),
    ])


def build_ann_pipeline(h: int, alpha: float, random_state: int = 0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(h,),
            alpha=alpha,
            max_iter=10000,
            early_stopping=True,
            n_iter_no_change=20,
            tol=1e-4,
            random_state=random_state,
        )),
    ])


# -------------------- Inner-CV helpers (error rate) --------------------

def inner_cv_error_rate(model: Pipeline, X: pd.DataFrame, y: np.ndarray,
                        n_splits: int = 5, random_state: int = 0) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc = cross_val_score(model, X, y, cv=kf, scoring="accuracy").mean()
    return float(1.0 - acc)


# -------------------- Nested CV --------------------

def nested_cv_classification(
    X: pd.DataFrame,
    y: np.ndarray,
    lambda_grid: Sequence[float],
    h_grid: Sequence[int],
    alpha_grid: Sequence[float],
    n_outer: int = 10,
    n_inner: int = 10,
    random_state: int = 0,
):
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    rows = []
    ann_records = []     # inner grid results for ANN
    logreg_records = []  # inner grid results for LogReg

    # For McNemar later
    pairwise_preds = []  # list of dicts with y_true, y_ann, y_logreg, y_base per outer fold

    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ---- Tune Logistic (λ*) ----
        best_lambda, best_logreg_cv = None, np.inf
        for lmbda in lambda_grid:
            model = build_logreg_pipeline(lmbda, random_state=random_state)
            er = inner_cv_error_rate(model, X_tr, y_tr, n_splits=n_inner, random_state=random_state)
            logreg_records.append({"outer_fold": fold_idx, "lambda": lmbda, "inner_cv_error": er})
            if er < best_logreg_cv:
                best_logreg_cv, best_lambda = er, lmbda

        # ---- Tune ANN (h*, α*) ----
        best_h, best_alpha, best_ann_cv = None, None, np.inf
        for h in h_grid:
            for a in alpha_grid:
                model = build_ann_pipeline(h, a, random_state=random_state)
                er = inner_cv_error_rate(model, X_tr, y_tr, n_splits=n_inner, random_state=random_state)
                ann_records.append({"outer_fold": fold_idx, "h": h, "alpha": a, "inner_cv_error": er})
                if er < best_ann_cv:
                    best_ann_cv, best_h, best_alpha = er, h, a

        # ---- Fit tuned models on outer-train and evaluate on outer-test ----
        logreg = build_logreg_pipeline(best_lambda, random_state=random_state).fit(X_tr, y_tr)
        ann = build_ann_pipeline(best_h, best_alpha, random_state=random_state).fit(X_tr, y_tr)

        y_hat_logreg = logreg.predict(X_te)
        y_hat_ann = ann.predict(X_te)

        err_logreg = float(1.0 - accuracy_score(y_te, y_hat_logreg))
        err_ann = float(1.0 - accuracy_score(y_te, y_hat_ann))

        # Baseline = majority class of outer-train
        maj = int(np.round(np.mean(y_tr)))  # argmax of {0,1} -> 1 if p>=0.5
        y_hat_base = np.full_like(y_te, maj)
        err_base = float(1.0 - accuracy_score(y_te, y_hat_base))

        rows.append({
            "outer_fold": fold_idx,
            "method2_h_star": best_h,
            "method2_alpha_star": best_alpha,
            "Etest_method2": err_ann,
            "lambda_star": best_lambda,
            "Etest_logistic": err_logreg,
            "Etest_baseline": err_base,
        })

        pairwise_preds.append({
            "y_true": y_te,
            "y_ann": y_hat_ann,
            "y_logreg": y_hat_logreg,
            "y_base": y_hat_base,
        })

        print(f"[fold {fold_idx}] ANN(h={best_h}, α={best_alpha}) err={err_ann:.3f} | "
              f"LogReg(λ={best_lambda}) err={err_logreg:.3f} | Base err={err_base:.3f}")

    # Save inner grids
    pd.DataFrame(ann_records).to_csv(RESULTS_DIR / "performance_ann_by_h_alpha.csv", index=False)
    pd.DataFrame(logreg_records).to_csv(RESULTS_DIR / "performance_logreg_by_lambda.csv", index=False)

    return rows, pairwise_preds


# -------------------- McNemar stats over outer folds --------------------

def mcnemar_from_preds(y_true, y_hat_A, y_hat_B):
    """Return chi2 (with continuity correction) and p-value for McNemar."""
    # Contingency counts
    b = int(np.sum((y_hat_A == y_true) & (y_hat_B != y_true)))  # A right, B wrong
    c = int(np.sum((y_hat_A != y_true) & (y_hat_B == y_true)))  # A wrong, B right
    if b + c == 0:
        return 0.0, 1.0
    stat = (fabs(b - c) - 1.0) ** 2 / (b + c)
    p = 1.0 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


def aggregate_mcnemar(pairwise_preds: List[dict]):
    """
    Pool all outer-test predictions and run McNemar for:
      (ANN vs LogReg), (ANN vs Baseline), (LogReg vs Baseline)
    """
    y_true = np.concatenate([d["y_true"] for d in pairwise_preds])
    y_ann = np.concatenate([d["y_ann"] for d in pairwise_preds])
    y_lr  = np.concatenate([d["y_logreg"] for d in pairwise_preds])
    y_b   = np.concatenate([d["y_base"] for d in pairwise_preds])

    out = {}
    out["ANN_vs_LogReg"] = mcnemar_from_preds(y_true, y_ann, y_lr)
    out["ANN_vs_Baseline"] = mcnemar_from_preds(y_true, y_ann, y_b)
    out["LogReg_vs_Baseline"] = mcnemar_from_preds(y_true, y_lr, y_b)
    return out


# -------------------- Final fit + artefacts --------------------

def fit_and_save_final_model(
    X_df: pd.DataFrame,
    y: np.ndarray,
    h_grid: Sequence[int],
    alpha_grid: Sequence[float],
    results_dir: Path = RESULTS_DIR,
    save_predictions: bool = False,
    random_state: int = 0,
):
    """Fit a final ANN on full data using inner-CV to pick (h*, α*), then persist artefacts."""
    best_h, best_a, best_cv = None, None, np.inf
    for h in h_grid:
        for a in alpha_grid:
            er = inner_cv_error_rate(build_ann_pipeline(h, a, random_state), X_df, y, n_splits=10, random_state=random_state)
            if er < best_cv:
                best_cv, best_h, best_a = er, h, a

    model = build_ann_pipeline(best_h, best_a, random_state).fit(X_df.values, y)

    result_dir = results_dir / SETUP_NAME
    result_dir.mkdir(parents=True, exist_ok=True)

    dump(model, result_dir / "model.joblib")
    (result_dir / "feature_names.json").write_text(json.dumps(list(X_df.columns), indent=2))
    (result_dir / "targets.json").write_text(json.dumps(["chd"], indent=2))

    if save_predictions:
        preds = model.predict(X_df.values)
        out_df = pd.DataFrame({"chd_actual": y, "chd_pred": preds}, index=X_df.index)
        out_df.to_csv(result_dir / "predictions.csv", index=True)

    return {"h_star": best_h, "alpha_star": best_a, "cv_error": best_cv, "model_path": str(result_dir / "model.joblib")}


def main():
    parser = argparse.ArgumentParser(description="Classification — nested CV (baseline vs logistic vs ANN)")
    parser.add_argument("--outer", type=int, default=10, help="K1 outer folds")
    parser.add_argument("--inner", type=int, default=10, help="K2 inner folds")
    parser.add_argument("--lambda-grid", nargs="+", type=float,
                        default=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                        help="λ values for logistic (C = 1/λ; λ=0 -> no reg)")
    parser.add_argument("--h-grid", nargs="+", type=int,
                        default=[1, 2, 4, 8, 12, 16, 24], help="Hidden units to try for ANN")
    parser.add_argument("--alpha-grid", nargs="+", type=float,
                        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], help="L2 α for ANN")
    parser.add_argument("--fit-final", action="store_true", help="Fit + save final ANN with CV-selected (h*, α*)")
    parser.add_argument("--save-predictions", action="store_true", help="When fitting final model, also save train predictions.csv")
    args = parser.parse_args()

    df = load_saheart()
    X_df, y = make_design_matrix_for_chd(df)

    rows, pairwise = nested_cv_classification(
        X=X_df, y=y,
        lambda_grid=args.lambda_grid,
        h_grid=args.h_grid, alpha_grid=args.alpha_grid,
        n_outer=args.outer, n_inner=args.inner, random_state=0
    )

    out = pd.DataFrame(rows, columns=[
        "outer_fold", "method2_h_star", "method2_alpha_star", "Etest_method2",
        "lambda_star", "Etest_logistic", "Etest_baseline"
    ])
    out_path = RESULTS_DIR / f"{SETUP_NAME}_nested_cv_classif.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\nTwo-level CV (Classification) — baseline vs logistic vs ANN")
    print(out.to_string(index=False))
    print(f"\nMean outer error — ANN: {out['Etest_method2'].mean():.4f} | "
          f"LogReg: {out['Etest_logistic'].mean():.4f} | "
          f"Baseline: {out['Etest_baseline'].mean():.4f}")
    print(f"[ok] Saved nested-CV table → {out_path}")

    # McNemar tests on pooled outer predictions
    stats = aggregate_mcnemar(pairwise)
    print("\nMcNemar tests (pooled across outer folds):")
    for k, (stat, p) in stats.items():
        print(f"  {k}: chi2={stat:.4f}, p={p:.4f}")

    if args.fit_final:
        summary = fit_and_save_final_model(
            X_df=X_df, y=y,
            h_grid=args.h_grid, alpha_grid=args.alpha_grid,
            results_dir=RESULTS_DIR,
            save_predictions=args.save_predictions,
            random_state=0,
        )
        print(f"\n[ok] Final ANN saved — h*={summary['h_star']}, α*={summary['alpha_star']}, "
              f"cv_error={summary['cv_error']:.4f}")
        print(f"Model path: {summary['model_path']}")


if __name__ == "__main__":
    main()
