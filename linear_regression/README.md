# Ridge Models for LDL and SBP

This directory hosts a ridge-regression study on the SAheart dataset. For each target
configuration (`ldl_only`, `sbp_only`, and `joint_ldl_sbp`) the script sweeps a grid of
regularisation strengths (λ), runs 10-fold cross-validation on the training split, and
evaluates the selected model on a 10% hold-out set to avoid information leakage.

## 1. Training Workflow

Run the full sweep with:

```bash
python -m linear_regression.run
```

Limit the run to specific targets:

```bash
python -m linear_regression.run --setting sbp_only
python -m linear_regression.run --setting ldl_only joint_ldl_sbp
```

Adjust the λ grid:

```bash
python -m linear_regression.run --lambda-min 1e-5 --lambda-max 1e2 --lambda-count 40
```

Add `--save-predictions` to persist train/test predictions. To score your own feature
matrix (already encoded like the training design matrix, e.g. including
`famhist_Present`), supply it via `--inference-input`. Predictions default to the
setting-specific results folder unless `--inference-output` is provided:

```bash
python -m linear_regression.run --setting sbp_only \
    --inference-input path/to/features.csv \
    --inference-output path/to/predictions.csv
```

## 2. Artefacts and Directory Layout

Outputs live in `linear_regression/results/<setting>/`:

- `lambda_curve.png` – mean CV MSE vs λ with the best λ highlighted.
- `cv_errors.csv` – per-fold MSE for every λ.
- `model.joblib` – fitted `StandardScaler + Ridge` pipeline.
- `feature_names.json` – ordered feature list required for inference.
- `targets.json` – names of the predicted target columns.
- `train_predictions.csv`, `test_predictions.csv` – optional predictions when `--save-predictions` is used.
- `custom_inference.csv` – optional predictions for user-supplied rows.

An aggregated `ridge_results.json` in `linear_regression/results/` captures the λ grid,
fold diagnostics, coefficients, intercepts, and both train/test MSE for every target
setup.

## 3. Manual Inference Playground

Use `inference.py` for quick experiments. Edit `TARGET_SETTING` and `CUSTOM_ROWS`, then
run:

```bash
python -m linear_regression.inference
```

The script loads the stored pipeline and writes the predictions to
`results/<setting>/manual_inference.csv` (unless you change `OUTPUT_PATH`).

## 4. Evaluation Summary

Generate consolidated figures and metric tables across regression and classification runs
with:

```bash
python -m analysis.visualize_results
```

By default the script writes figures and CSV summaries to `analysis/output/`.
