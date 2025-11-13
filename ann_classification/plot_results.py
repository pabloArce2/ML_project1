import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def plot_outer_errors(tbl, out_dir):
    df = pd.read_csv(tbl)
    means = df["Etest_ann"].mean()
    stds = df["Etest_ann"].std(ddof=1)

    plt.figure(figsize=(5, 4))
    plt.bar(["ANN"], [means], yerr=[stds], capsize=8)
    plt.ylabel("Error rate (1 - accuracy)")
    plt.title("ANN outer-fold mean error +/- std")
    plt.tight_layout()
    plt.savefig(out_dir / "ann_outer_errors.png", dpi=160)
    plt.close()


def plot_ann_grid(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    if "h" in df.columns:
        h_col = "h"
    elif "h_base" in df.columns:
        h_col = "h_base"
    else:
        raise KeyError("Expected either 'h' or 'h_base' in performance table.")

    pivot = df.pivot_table(index=h_col, columns="alpha",
                           values="inner_cv_error", aggfunc="mean").sort_index()

    plt.figure(figsize=(7, 5))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label="Inner CV error")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.xlabel("alpha")
    plt.ylabel(h_col)
    plt.title("ANN: inner CV heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "ann_heatmap.png", dpi=160)
    plt.close()


def plot_accuracy_vs_h(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(df["h"], df["best_inner_accuracy"], marker="o")
    plt.xlabel("Hidden units (h)")
    plt.ylabel("Best inner-CV accuracy")
    plt.ylim(0.4, 1.0)
    plt.grid(alpha=0.4)
    plt.title("ANN accuracy vs hidden units")
    plt.tight_layout()
    plt.savefig(out_dir / "ann_accuracy_vs_h.png", dpi=160)
    plt.close()


def plot_final_ann(folder):
    pred_path = folder / "predictions.csv"
    if not pred_path.exists():
        return

    df = pd.read_csv(pred_path)
    y_true = df["chd_actual"].values
    y_pred = df["chd_pred"].values

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d")
    plt.title("Final ANN - Confusion Matrix")
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

    acc_vs_h = out / "ann_accuracy_vs_h.csv"
    if acc_vs_h.exists():
        plot_accuracy_vs_h(acc_vs_h, out)

    final = out / args.setting
    if final.exists():
        plot_final_ann(final)

    print("[OK] ANN plots saved.")


if __name__ == "__main__":
    main()
