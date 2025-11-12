# Logistic Regression Classification

This module mirrors the regression tooling and trains an L2-regularised logistic
regression model to predict the binary `chd` outcome in the SAheart dataset. A logarithmic
grid of λ values (default 60 between `1e-5` and `1e4`) is evaluated with 10-fold
cross-validation on the training split, while a 10% hold-out set is kept for final testing
to prevent leakage.

## Usage

```bash
python -m logistic_regression.run
```

Common flags:

- `--folds` – Number of stratified CV folds (default `10`).
- `--lambda-min`, `--lambda-max`, `--lambda-count` – Control the λ grid.
- `--random-state` – Seed for the train/test split and CV shuffling (default `42`).
- `--save-predictions` – Store train/test predictions for further analysis.
- `--inference-input` / `--inference-output` – Score custom feature rows with the fitted model.

## Outputs

Results are written to `logistic_regression/results/logistic_regression/`:

- `cv_metrics.csv` – Per-fold accuracy and log-loss for every λ.
- `accuracy_curve.png` – CV accuracy vs λ.
- `model.joblib` – Serialised `StandardScaler + LogisticRegression` pipeline.
- `feature_names.json` – Ordered feature list used during training.
- `targets.json` – Identifies the predicted label (`chd`).
- `train_predictions.csv`, `test_predictions.csv` – Optional predictions when `--save-predictions` is supplied.
- `custom_inference.csv` – Optional predictions for user-supplied rows.
- `summary.json` – Metadata for the selected λ, including CV, train, and test metrics.

The aggregated `classification_results.json` summarises each run, capturing the λ grid,
cross-validation diagnostics, and held-out test scores.

## Evaluation Summary

After running the experiments, call:

```bash
python -m analysis.visualize_results
```

to generate consolidated figures and tables (written to `analysis/output/`) covering both
the regression and classification pipelines.
