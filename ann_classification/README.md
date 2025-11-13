# ANN Classification (`chd`)

Implements binary classification for the SAHeart dataset using an
**Artificial Neural Network (ANN)**. The model tunes:

- Number of hidden units _h_
- L2 regularization _α_

Evaluation is done with **nested cross-validation**, using **error rate**
(1 − accuracy) as the metric.

---

## How to run

### 1) Nested CV (main experiment)

```bash
python -m ann_classification.run --outer 10 --inner 10
```

This writes the main results table and the inner-CV grid.

### 2) Optional: also save a final ANN trained on all data

```bash
python -m ann_classification.run --outer 10 --inner 10 --fit-final
```

This additionally trains a final ANN on the full dataset using the
best (h, α) from inner CV and saves it for inference.

### 3) Predict on new samples

```bash
python -m ann_classification.inference --input path/to/your_rows.csv --output preds.csv
```

This loads the saved ANN and adds `chd_pred` (and optionally `chd_prob`)
columns to the given CSV.

---

## Outputs

### Under `ann_classification/results/`

- `chd_nested_cv_ann.csv` — outer-fold error rates and selected (h*, α*).
- `performance_ann_by_h_alpha.csv` — inner-CV search over ANN hyperparameters.
- `ann_outer_errors.png` — bar plot of mean outer error ± std for the ANN.
- `ann_heatmap.png` — heatmap of inner-CV error over (h, α).

### Under `ann_classification/results/chd/` (created if `--fit-final` is used)

- `model.joblib` — final ANN model (scaler + MLP).
- `feature_names.json` — encoded input columns used by the model.
- `targets.json` — target name (`["chd"]`).
- `predictions.csv` — full-dataset `chd_actual` vs `chd_pred`.
- `confusion_matrix_ann.png` — confusion matrix of the final ANN.
