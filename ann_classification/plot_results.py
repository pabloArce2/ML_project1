import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

def plot_outer_errors(tbl, out_dir):
    df = pd.read_csv(tbl)
    means = df["Etest_ann"].mean()
    stds = df["Etest_ann"].std(ddof=1)

    plt.figure(figsize=(5,4))
    plt.bar(["ANN"], [means], yerr=[stds], capsize=8)
    plt.ylabel("Error rate (1 - accuracy)")
    plt.title("ANN — Outer-fold mean error ± std")
    plt.tight_layout()
    plt.savefig(out_dir / "ann_outer_errors.png", dpi=160)
    plt.close()

def plot_ann_grid(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index="h", columns="alpha",
                           values="inner_cv_error", aggfunc="mean")

    plt.figure(figsize=(7,5))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label="Inner CV error")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.xlabel("alpha")
    plt.ylabel("h")
    plt.title("ANN: inner CV heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "ann_heatmap.png", dpi=160)
    plt.close()

def plot_final_ann(folder):
    pred_path = folder / "predictions.csv"
    if not pred_path.exists():
        return

    df = pd.read_csv(pred_path)
    y_true = df["chd_actual"].values
    y_pred = df["chd_pred"].values

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d")
    plt.title("Final ANN — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(folder / "confusion_matrix_ann.png", dpi=160)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=RES)
    p.add_argument("--setting", default="chd")
    args = p.parse_args()

    out = Path(args.results_dir)
    tbl = out / "chd_nested_cv_ann.csv"
    if tbl.exists():
        plot_outer_errors(tbl, out)

    ann_grid = out / "performance_ann_by_h_alpha.csv"
    if ann_grid.exists():
        plot_ann_grid(ann_grid, out)

    final = out / args.setting
    if final.exists():
        plot_final_ann(final)

    print("[OK] ANN plots saved.")

if __name__ == "__main__":
    main()
