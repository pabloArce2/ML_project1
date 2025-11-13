import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"
SETUP_NAME = "chd"


def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    df["famhist"] = df["famhist"].astype("category")
    return df


def make_design(df: pd.DataFrame):
    X_df = pd.get_dummies(df.drop(columns=["chd"]), drop_first=True, dtype=float)
    y = df["chd"].astype(int).values
    return X_df, y


def _parse_layer_pattern(pattern: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in pattern.split(",") if p.strip()]
    if not parts:
        return (1.0,)
    return tuple(float(p) for p in parts)


def expand_layers(h: int, pattern: str) -> Tuple[int, ...]:
    ratios = _parse_layer_pattern(pattern)
    layers = [max(1, int(round(h * ratio))) for ratio in ratios]
    return tuple(layers)


def format_layers(layers: Tuple[int, ...]) -> str:
    return "x".join(str(v) for v in layers)


def build_ann(hidden_layers: Tuple[int, ...], alpha: float, solver: str, random_state=0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            alpha=alpha,
            solver=solver,
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


def generate_configs(h_grid: Iterable[int], layer_patterns: Iterable[str],
                     alpha_grid: Iterable[float]) -> Iterable[Dict]:
    for h in h_grid:
        for pattern in layer_patterns:
            layers = expand_layers(h, pattern)
            for alpha in alpha_grid:
                yield {
                    "h_base": h,
                    "layer_pattern": pattern,
                    "hidden_layers": layers,
                    "hidden_layers_str": format_layers(layers),
                    "alpha": alpha,
                }


def nested_cv_ann(X, y, h_grid, layer_patterns, alpha_grid,
                  solver, K1=10, K2=10):
    outer = KFold(n_splits=K1, shuffle=True, random_state=0)
    rows: List[Dict] = []
    ann_records: List[Dict] = []
    outer_preds: List[Dict] = []

    for fold, (tr, te) in enumerate(outer.split(X), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        best_cfg: Dict = {}
        best_cv = np.inf

        for cfg in generate_configs(h_grid, layer_patterns, alpha_grid):
            model = build_ann(cfg["hidden_layers"], cfg["alpha"], solver)
            err = inner_cv_error(model, Xtr, ytr, K2)
            record = {
                "outer_fold": fold,
                "h_base": cfg["h_base"],
                "layer_pattern": cfg["layer_pattern"],
                "hidden_layers": cfg["hidden_layers_str"],
                "alpha": cfg["alpha"],
                "inner_cv_error": err,
                "inner_cv_accuracy": 1 - err,
            }
            ann_records.append(record)
            if err < best_cv:
                best_cv = err
                best_cfg = cfg.copy()

        final = build_ann(best_cfg["hidden_layers"],
                          best_cfg["alpha"],
                          solver).fit(Xtr, ytr)
        yhat = final.predict(Xte)
        err_te = 1 - accuracy_score(yte, yhat)
        prob_te = None
        if hasattr(final, "predict_proba"):
            prob_te = final.predict_proba(Xte)[:, 1]

        rows.append({
            "outer_fold": fold,
            "h_star": best_cfg["h_base"],
            "layer_pattern_star": best_cfg["layer_pattern"],
            "hidden_layers_star": best_cfg["hidden_layers_str"],
            "alpha_star": best_cfg["alpha"],
            "Etest_ann": err_te,
            "accuracy_ann": 1 - err_te,
        })

        if prob_te is None:
            prob_te = np.full_like(yhat, np.nan, dtype=float)

        for sample_idx, y_true, y_pred, y_prob in zip(Xte.index, yte, yhat, prob_te):
            outer_preds.append({
                "outer_fold": fold,
                "sample_index": int(sample_idx),
                "y_true": int(y_true),
                "y_pred": int(y_pred),
                "y_prob": float(y_prob),
                "h_star": best_cfg["h_base"],
                "layer_pattern_star": best_cfg["layer_pattern"],
                "hidden_layers_star": best_cfg["hidden_layers_str"],
                "alpha_star": best_cfg["alpha"],
            })

        print(f"[fold {fold}] ANN(layers={best_cfg['hidden_layers_str']}, "
              f"alpha={best_cfg['alpha']}) -> acc={1 - err_te:.3f}")

    return rows, ann_records, outer_preds


def fit_final(X, y, h_grid, layer_patterns, alpha_grid, solver):
    best_cfg: Dict = {}
    best_cv = np.inf
    for cfg in generate_configs(h_grid, layer_patterns, alpha_grid):
        model = build_ann(cfg["hidden_layers"], cfg["alpha"], solver)
        err = inner_cv_error(model, X, y)
        if err < best_cv:
            best_cv = err
            best_cfg = cfg.copy()

    final = build_ann(best_cfg["hidden_layers"],
                      best_cfg["alpha"],
                      solver).fit(X, y)
    out_dir = RESULTS_DIR / SETUP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    dump(final, out_dir / "model.joblib")
    (out_dir / "feature_names.json").write_text(json.dumps(list(X.columns), indent=2))
    (out_dir / "targets.json").write_text(json.dumps(["chd"], indent=2))

    preds = final.predict(X)
    df = pd.DataFrame({"chd_actual": y, "chd_pred": preds})
    if hasattr(final, "predict_proba"):
        probs = final.predict_proba(X)[:, 1]
        df["chd_prob"] = probs
    df.to_csv(out_dir / "predictions.csv", index=False)

    return best_cfg


def main():
    parser = argparse.ArgumentParser(description="ANN-only classification")
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--h-grid", nargs="+", type=int, default=[12, 16, 22, 24])
    parser.add_argument("--layer-patterns", nargs="+", default=["1"],
                        help="Comma-separated ratios applied to each base h (e.g. '1,0.5').")
    parser.add_argument("--alpha-grid", nargs="+", type=float,
                        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0])
    parser.add_argument("--fit-final", action="store_true")
    parser.add_argument("--solver", default="adam",
                        choices=["adam", "lbfgs", "sgd"],
                        help="MLP solver to use (default: adam)")
    args = parser.parse_args()

    df = load_saheart()
    X, y = make_design(df)

    rows, ann_grid, outer_preds = nested_cv_ann(
        X, y,
        args.h_grid,
        args.layer_patterns,
        args.alpha_grid,
        args.solver,
        args.outer,
        args.inner,
    )

    ann_df = pd.DataFrame(rows)
    ann_df.to_csv(RESULTS_DIR / "chd_nested_cv_ann.csv", index=False)

    ann_grid_df = pd.DataFrame(ann_grid)
    ann_grid_df.to_csv(RESULTS_DIR / "performance_ann_by_h_alpha.csv", index=False)

    pd.DataFrame(outer_preds).to_csv(RESULTS_DIR / "chd_outer_predictions.csv", index=False)

    acc_by_h = (ann_grid_df
                .sort_values(["h_base", "inner_cv_error"])
                .groupby("h_base", as_index=False)
                .first()[["h_base", "inner_cv_accuracy"]])
    acc_by_h.rename(columns={"h_base": "h", "inner_cv_accuracy": "best_inner_accuracy"}, inplace=True)
    acc_by_h.to_csv(RESULTS_DIR / "ann_accuracy_vs_h.csv", index=False)

    if args.fit_final:
        cfg = fit_final(X, y,
                        args.h_grid,
                        args.layer_patterns,
                        args.alpha_grid,
                        args.solver)
        print(f"[OK] Final ANN saved (layers={cfg['hidden_layers_str']}, "
              f"alpha={cfg['alpha']}) using solver={args.solver}")


if __name__ == "__main__":
    main()
