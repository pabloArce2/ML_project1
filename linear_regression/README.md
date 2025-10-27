# Ridge Models for LDL and SBP

This directory hosts a ridge-regression study on the SAheart dataset. The script sweeps a
grid of regularisation strengths (λ) to predict `ldl`, `sbp`, and the joint vector
`[ldl, sbp]` from the remaining risk factors, logging diagnostics and artefacts for each
target configuration.

## 1. Targets, Predictors, and Feature Engineering

- **LDL-only model** – Predicts `ldl` from `sbp`, `tobacco`, `adiposity`, `famhist`,
  `typea`, `obesity`, `alcohol`, and `age`. `sbp` remains a predictor because we only
  remove the response column.
- **SBP-only model** – Predicts `sbp` from the same covariates, with `ldl` retained as an
  explanatory variable.
- **Joint model** – Predicts both `ldl` and `sbp` simultaneously; both targets are removed
  from the design matrix and used exclusively as ground-truth outputs.

The categorical attribute `famhist` is expanded with one-of-K coding (baseline dropped)
and every numeric column is standardised before fitting the ridge model.

## 2. Training Workflow

Run the full sweep with:

```bash
python -m linear_regression.run
```

Limit the run to specific targets:

```bash
python -m linear_regression.run --setting sbp_only
python -m linear_regression.run --setting ldl_only joint_ldl_sbp
```

Adjust the λ grid as needed:

```bash
python -m linear_regression.run --lambda-min 1e-5 --lambda-max 1e2 --lambda-count 40
```

Use `--save-predictions` to persist training predictions. To score your own feature
matrix (already encoded like the training design matrix, e.g. including
`famhist_Present`), supply it via `--inference-input`. The predictions default to the
setting’s results folder unless `--inference-output` is provided:

```bash
python -m linear_regression.run --setting sbp_only \
    --inference-input path/to/features.csv \
    --inference-output path/to/predictions.csv
```

## 3. Artefacts and Directory Layout

All outputs are stored under `linear_regression/results/<setting>/`. Each setting folder
contains:

- `lambda_curve.png` – CV MSE vs λ, annotated with the best λ.
- `cv_errors.csv` – Per-fold MSE for every λ on the grid.
- `model.joblib` – The fitted `StandardScaler + Ridge` pipeline.
- `feature_names.json` – Ordered feature list required for inference.
- `targets.json` – Names of the predicted target columns.
- `predictions.csv` – Optional training predictions (`--save-predictions`).
- `custom_inference.csv` – Optional predictions for user-supplied rows.

An aggregated `ridge_results.json` is written to `linear_regression/results/` with the λ
grid, mean errors, per-fold breakdowns, coefficients, and intercepts for each setting.

## 4. Manual Inference Playground

Use `inference.py` for quick experiments. Edit `TARGET_SETTING` and `CUSTOM_ROWS`, then
run:

```bash
python -m linear_regression.inference
```

The script loads the stored pipeline and writes the predictions to
`results/<setting>/manual_inference.csv` (unless you change `OUTPUT_PATH`), so you can
iterate on scenarios without touching the training driver.
