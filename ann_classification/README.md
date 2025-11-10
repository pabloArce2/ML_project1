# ANN Classification (`chd`)

Implements binary classification for the SAHeart dataset using:

- **Baseline** (majority class)
- **Logistic Regression** (regularized, tuning λ)
- **Artificial Neural Network (ANN)** (tuning hidden units _h_ and regularization _α_)

All models are evaluated using **nested cross-validation** with **error rate** as the metric.

## How to run

### 1) Nested CV (main experiment)

```bash
python -m ann_classification.run \
  --outer 10 --inner 10
```

### Optional: also save a final ANN trained on all data

```bash
python -m ann_classification.run \
  --outer 10 --inner 10 \
  --fit-final --save-predictions
```

### 2) Predict on new samples

```bash
python -m ann_classification.inference \
  --input path/to/your_rows.csv \
  --output preds.csv
```

## Outputs (saved under `ann_classification/results/chd_only/`)

- `chd_only_nested_cv_classif.csv` — outer-fold error rates for all three methods
- `performance_logreg_by_lambda.csv` — inner search over λ
- `performance_ann_by_h_alpha.csv` — inner search over ANN hyperparameters
- `model.joblib` — final ANN (if `--fit-final`)
- `feature_names.json` — encoded input columns
- `predictions.csv` — full-dataset predictions (if `--save-predictions`)
