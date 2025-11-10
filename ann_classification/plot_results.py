# ann_classification/plot_results.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

def plot_outer_errors(table_path: Path, out_dir: Path):
    df = pd.read_csv(table_path)
    # Means + stds
    means = {
        "Baseline": df["Etest_baseline"].mean(),
        "Logistic": df["Etest_logistic"].mean(),
        "ANN": df["Etest_method2"].mean(),
    }
    stds = {
        "Baseline": df["Etest_baseline"].std(ddof=1),
        "Logistic": df["Etest_logistic"].std(ddof=1),
        "ANN": df["Etest_method2"].std(ddof=1),
    }

    # Bar chart with error bars
    labels = list(means.keys())
    vals = [means[k] for k in labels]
    errs = [stds[k] for k in labels]

    plt.figure(figsize=(6,4))
    plt.bar(labels, vals, yerr=errs, capsize=6)
    plt.ylabel("Error rate (1 - accuracy)")
    plt.title("Outer-fold mean error ± std")
    plt.tight_layout()
    plt.savefig(out_dir / "outer_errors_bar.png", dpi=160)
    plt.close()

    # Per-fold lines
    plt.figure(figsize=(7,4))
    x = df["outer_fold"].values
    plt.plot(x, df["Etest_baseline"].values, marker="o", label="Baseline")
    plt.plot(x, df["Etest_logistic"].values, marker="o", label="Logistic")
    plt.plot(x, df["Etest_method2"].values, marker="o", label="ANN")
    plt.xlabel("Outer fold")
    plt.ylabel("Error rate")
    plt.title("Per-fold error (outer test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "outer_errors_per_fold.png", dpi=160)
    plt.close()

def plot_logreg_lambda(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    g = df.groupby("lambda", as_index=False)["inner_cv_error"].mean().sort_values("lambda")
    plt.figure(figsize=(6,4))
    plt.plot(g["lambda"], g["inner_cv_error"], marker="o")
    plt.xscale("log")
    plt.xlabel("λ (log scale)")
    plt.ylabel("Inner-CV error")
    plt.title("Logistic: inner-CV error vs λ")
    plt.tight_layout()
    plt.savefig(out_dir / "performance_logreg_by_lambda.png", dpi=160)
    plt.close()

def plot_ann_heatmap(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index="h", columns="alpha", values="inner_cv_error", aggfunc="mean")
    h_vals = pivot.index.values
    a_vals = pivot.columns.values
    Z = pivot.values

    plt.figure(figsize=(7,5))
    im = plt.imshow(Z, aspect="auto", origin="lower",
                    extent=[np.log10(a_vals.min()), np.log10(a_vals.max()), h_vals.min()-0.5, h_vals.max()+0.5])
    plt.colorbar(im, label="Inner-CV error")
    plt.yticks(h_vals)
    # Show α ticks in original scale but log10 on axis
    aticks = np.unique(a_vals)
    plt.xticks(np.log10(aticks), [f"{a:g}" for a in aticks], rotation=0)
    plt.xlabel("α")
    plt.ylabel("h (hidden units)")
    plt.title("ANN: inner-CV error heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "performance_ann_by_h_alpha.png", dpi=160)
    plt.close()

def plot_final_ann_diagnostics(result_folder: Path):
    pred_path = result_folder / "predictions.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path)
    if "chd_actual" not in df.columns or ("chd_prob" not in df.columns and "chd_pred" not in df.columns):
        return

    y_true = df["chd_actual"].values.astype(int)

    # ROC (needs probability)
    if "chd_prob" in df.columns:
        y_prob = df["chd_prob"].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Final ANN — ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(result_folder / "roc_final_ann.png", dpi=160)
        plt.close()

    # Confusion matrix (uses predicted labels)
    y_pred = df["chd_pred"].values.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot(values_format="d")
    plt.title("Final ANN — Confusion matrix")
    plt.tight_layout()
    plt.savefig(result_folder / "confusion_matrix_final_ann.png", dpi=160)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Make plots from classification results")
    parser.add_argument("--results-dir", type=Path, default=RES, help="ann_classification/results")
    parser.add_argument("--setting", default="chd_only")
    args = parser.parse_args()

    out_dir = args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Outer-fold errors
    tbl = out_dir / f"{args.setting}_nested_cv_classif.csv"
    if tbl.exists():
        plot_outer_errors(tbl, out_dir)

    # 2) Logistic grid
    logreg_csv = out_dir / "performance_logreg_by_lambda.csv"
    if logreg_csv.exists():
        plot_logreg_lambda(logreg_csv, out_dir)

    # 3) ANN grid
    ann_csv = out_dir / "performance_ann_by_h_alpha.csv"
    if ann_csv.exists():
        plot_ann_heatmap(ann_csv, out_dir)

    # 4) Final ANN diagnostics
    final_folder = out_dir / args.setting
    if final_folder.exists():
        plot_final_ann_diagnostics(final_folder)

    print(f"[ok] Plots saved to: {out_dir}")

if __name__ == "__main__":
    main()
