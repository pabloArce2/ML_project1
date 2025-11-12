import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RIDGE_RESULTS_PATH = BASE_DIR / "linear_regression" / "results" / "ridge_results.json"
CLASS_RESULTS_PATH = BASE_DIR / "logistic_regression" / "results" / "classification_results.json"


def _load_json(path: Path) -> Optional[List[dict]]:
    if not path.exists():
        print(f"[visualize_results] Missing file: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_regression_summary(output_dir: Path) -> None:
    data = _load_json(RIDGE_RESULTS_PATH)
    if not data:
        print("[visualize_results] No ridge results found; skipping regression plots.")
        return

    df_rows = []
    n_targets = len(data)
    fig, axes = plt.subplots(
        2,
        n_targets,
        figsize=(4.5 * n_targets, 7),
        squeeze=False,
        sharex="col",
    )

    for idx, entry in enumerate(data):
        lambdas = np.asarray(entry.get("lambda_grid", []), dtype=float)
        cv_errors = np.asarray(entry.get("mean_cv_errors", []), dtype=float)
        target = entry.get("target", f"target_{idx}")
        best_lambda = entry.get("best_lambda", np.nan)
        best_error = entry.get("best_error", np.nan)
        train_error = entry.get("train_error")
        test_error = entry.get("test_error")

        ax_curve = axes[0, idx]
        if lambdas.size > 0 and cv_errors.size == lambdas.size:
            ax_curve.plot(lambdas, cv_errors, marker="o", linewidth=1.6)
            ax_curve.set_xscale("log")
            if not np.isnan(best_lambda):
                ax_curve.axvline(best_lambda, color="crimson", linestyle="--", linewidth=1.2)
                ax_curve.scatter([best_lambda], [best_error], color="crimson", zorder=5)
        ax_curve.set_title(f"{target}\nCV MSE vs λ")
        ax_curve.set_xlabel("λ")
        ax_curve.set_ylabel("CV MSE")
        ax_curve.grid(True, which="both", linestyle="--", alpha=0.3)

        ax_bar = axes[1, idx]
        bar_labels = ["Train", "CV (best)", "Test"]
        raw_values = [train_error, best_error, test_error]
        colors = ["#4C72B0", "#55A868", "#C44E52"]

        safe_values = []
        annotations = []
        for value in raw_values:
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                safe_values.append(0.0)
                annotations.append("N/A")
            else:
                safe_values.append(value)
                annotations.append(f"{value:.3f}")

        ax_bar.bar(bar_labels, safe_values, color=colors)
        ax_bar.set_title(f"{target}\nError comparison")
        ax_bar.set_ylabel("MSE")
        for label, value, annotation in zip(bar_labels, safe_values, annotations):
            ax_bar.text(
                label,
                value,
                annotation,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        df_rows.append(
            {
                "target": target,
                "best_lambda": best_lambda,
                "cv_mse": best_error,
                "train_mse": train_error,
                "test_mse": test_error,
            }
        )

    plt.tight_layout()
    output_path = output_dir / "ridge_summary.png"
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[visualize_results] Saved regression summary figure to {output_path}")

    summary_df = pd.DataFrame(df_rows)
    summary_df.to_csv(output_dir / "ridge_summary_metrics.csv", index=False)
    print(f"[visualize_results] Saved regression summary metrics to ridge_summary_metrics.csv")


def plot_classification_summary(output_dir: Path) -> None:
    data = _load_json(CLASS_RESULTS_PATH)
    if not data:
        print("[visualize_results] No classification results found; skipping classification plots.")
        return

    df_rows = []
    for entry in data:
        lambdas = np.asarray(entry.get("param_grid", []), dtype=float)
        mean_acc = np.asarray(entry.get("mean_accuracy", []), dtype=float)
        mean_loss = np.asarray(entry.get("mean_log_loss", []), dtype=float)
        best_param = entry.get("best_param", np.nan)
        best_acc = entry.get("best_accuracy", np.nan)
        best_loss = entry.get("best_log_loss", np.nan)
        train_acc = entry.get("train_accuracy")
        train_loss = entry.get("train_log_loss")
        test_acc = entry.get("test_accuracy")
        test_loss = entry.get("test_log_loss")

        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        if lambdas.size > 0 and mean_acc.size == lambdas.size:
            axes[0].plot(lambdas, mean_acc, marker="o", linewidth=1.6)
            axes[0].set_xscale("log")
            axes[0].axvline(best_param, color="crimson", linestyle="--", linewidth=1.2)
            axes[0].scatter([best_param], [best_acc], color="crimson", zorder=5)
        axes[0].set_title("CV Accuracy vs λ")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(True, which="both", linestyle="--", alpha=0.3)

        if lambdas.size > 0 and mean_loss.size == lambdas.size:
            axes[1].plot(lambdas, mean_loss, marker="o", linewidth=1.6, color="#C44E52")
            axes[1].set_xscale("log")
            axes[1].axvline(best_param, color="crimson", linestyle="--", linewidth=1.2)
            axes[1].scatter([best_param], [best_loss], color="crimson", zorder=5)
        axes[1].set_title("CV Log-loss vs λ")
        axes[1].set_xlabel("λ")
        axes[1].set_ylabel("Log-loss")
        axes[1].grid(True, which="both", linestyle="--", alpha=0.3)

        plt.tight_layout()
        classification_curve_path = output_dir / "classification_curves.png"
        plt.savefig(classification_curve_path, dpi=200)
        plt.close(fig)
        print(f"[visualize_results] Saved classification curves to {classification_curve_path}")

        # Train/test bars
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        categories = ["Train", "CV (best)", "Test"]
        acc_values = [train_acc, best_acc, test_acc]
        ax_bar.bar(categories, acc_values, color="#4C72B0")
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Accuracy")
        ax_bar.set_title("Train / CV / Test accuracy")
        for cat, val in zip(categories, acc_values):
            if val is not None and not np.isnan(val):
                ax_bar.text(cat, val, f"{val:.3f}", ha="center", va="bottom")
        accuracy_bar_path = output_dir / "classification_accuracy_breakdown.png"
        plt.tight_layout()
        plt.savefig(accuracy_bar_path, dpi=200)
        plt.close(fig_bar)
        print(f"[visualize_results] Saved classification accuracy comparison to {accuracy_bar_path}")

        df_rows.append(
            {
                "best_lambda": best_param,
                "cv_accuracy": best_acc,
                "cv_log_loss": best_loss,
                "train_accuracy": train_acc,
                "train_log_loss": train_loss,
                "test_accuracy": test_acc,
                "test_log_loss": test_loss,
            }
        )

    summary_df = pd.DataFrame(df_rows)
    summary_df.to_csv(output_dir / "classification_summary_metrics.csv", index=False)
    print("[visualize_results] Saved classification summary metrics to classification_summary_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise regression and classification experiment results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "output",
        help="Directory where summary figures and tables should be written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    _ensure_dir(output_dir)

    plot_regression_summary(output_dir)
    plot_classification_summary(output_dir)


if __name__ == "__main__":
    main()
