import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"
SETUP_NAME = "chd"  # cleaner name than chd_only


def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    df["famhist"] = df["famhist"].astype("category")
    return df


def make_design(df: pd.DataFrame):
    X_df = pd.get_dummies(df.drop(columns=["chd"]), drop_first=True, dtype=float)
    y = df["chd"].astype(int).values
    return X_df, y


def build_ann(h: int, alpha: float, random_state=0):
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
        ))
    ])


def inner_cv_error(model, X, y, K=10):
    acc = cross_val_score(model, X, y, cv=K, scoring="accuracy").mean()
    return 1 - acc


def nested_cv_ann(X, y, h_grid, alpha_grid, K1=10, K2=10):
    outer = KFold(n_splits=K1, shuffle=True, random_state=0)
    rows, ann_records = [], []

    for fold, (tr, te) in enumerate(outer.split(X), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        best_h, best_a, best_cv = None, None, np.inf

        for h in h_grid:
            for a in alpha_grid:
                model = build_ann(h, a)
                err = inner_cv_error(model, Xtr, ytr, K2)
                ann_records.append({"outer_fold": fold, "h": h, "alpha": a, "inner_cv_error": err})
                if err < best_cv:
                    best_cv, best_h, best_a = err, h, a

        final = build_ann(best_h, best_a).fit(Xtr, ytr)
        yhat = final.predict(Xte)
        err_te = 1 - accuracy_score(yte, yhat)

        rows.append({
            "outer_fold": fold,
            "h_star": best_h,
            "alpha_star": best_a,
            "Etest_ann": err_te,
        })

        print(f"[fold {fold}] ANN(h={best_h}, a={best_a}) → error={err_te:.3f}")

    return rows, ann_records


def fit_final(X, y, h_grid, alpha_grid):
    best_h, best_a, best_cv = None, None, np.inf
    for h in h_grid:
        for a in alpha_grid:
            err = inner_cv_error(build_ann(h, a), X, y)
            if err < best_cv:
                best_cv, best_h, best_a = err, h, a

    model = build_ann(best_h, best_a).fit(X.values, y)
    out_dir = RESULTS_DIR / SETUP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    dump(model, out_dir / "model.joblib")
    (out_dir / "feature_names.json").write_text(json.dumps(list(X.columns), indent=2))
    (out_dir / "targets.json").write_text(json.dumps(["chd"], indent=2))

    preds = model.predict(X.values)
    df = pd.DataFrame({"chd_actual": y, "chd_pred": preds})
    df.to_csv(out_dir / "predictions.csv", index=False)

    return best_h, best_a


def main():
    parser = argparse.ArgumentParser(description="ANN-only classification")
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--h-grid", nargs="+", type=int, default=[1,2,4,8,12,16,24])
    parser.add_argument("--alpha-grid", nargs="+", type=float,
                        default=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])
    parser.add_argument("--fit-final", action="store_true")
    args = parser.parse_args()

    df = load_saheart()
    X, y = make_design(df)

    rows, ann_grid = nested_cv_ann(X, y, args.h_grid, args.alpha_grid,
                                   args.outer, args.inner)

    ann_df = pd.DataFrame(rows)
    ann_df.to_csv(RESULTS_DIR / "chd_nested_cv_ann.csv", index=False)

    pd.DataFrame(ann_grid).to_csv(RESULTS_DIR / "performance_ann_by_h_alpha.csv", index=False)

    if args.fit_final:
        h_star, a_star = fit_final(X, y, args.h_grid, args.alpha_grid)
        print(f"[OK] Final ANN saved (h*={h_star}, α*={a_star})")


if __name__ == "__main__":
    main()
