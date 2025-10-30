# Logistic Regression Classification

This module mirrors the linear regression tooling and trains an L2-regularised logistic regression model to predict the binary `chd` outcome in the SAheart dataset. The sole complexity parameter is the regularisation strength `λ`, explored on a logarithmic grid (default 60 values between `1e-5` and `1e4`). Each `λ` value maps to scikit-learn's `C = 1 / λ`.

## Usage

```bash
python -m logistic_regression.run
```

Common flags:

- `--folds` – Number of stratified CV folds (default `10`).
- `--lambda-min`, `--lambda-max`, `--lambda-count` – Control the `λ` grid for logistic regression (default `1e-5` to `1e4` with 60 points).
- `--save-predictions` – Store train-set predictions for further analysis.
- `--inference-input` – CSV with feature columns to score using the chosen model (paired with `--inference-output`).

Outputs (saved under `logistic_regression/results/logistic_regression/`):

- Cross-validation diagnostics (`cv_metrics.csv`, accuracy curve plot).
- Serialised model pipeline (`model.joblib`) and feature ordering (`feature_names.json`).
- Metadata with the selected parameter, accuracy metrics, and description (`summary.json`).
