# Artificial Neural Network Regression (Project 2 — Part b)

This module implements the **ANN regression component** of the Project 2 assignment.  
It trains shallow feed-forward neural networks on the **South African Heart Disease dataset** and evaluates them using **nested cross-validation**.

---

## Folder structure

```
ann_regression/
├── run.py                 # Main training script (nested CV + final model fitting)
├── inference.py           # Run predictions using a saved model
└── results/               # Generated models, predictions, and evaluation plots
```

---

## Requirements

This module relies on the following key Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `joblib`

---

## Running training and evaluation

The main entry point is `run.py`.  
It performs **two-level cross-validation** and optionally fits and saves a final model.

### LDL prediction example

```bash
python -m ann_regression.run   --setting ldl_only   --outer 10 --inner 10   --h-grid 1 2 4 8 12 16 20 24   --alpha-grid 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10 100   --fit-final --save-predictions
```

### SBP prediction example

```bash
python -m ann_regression.run   --setting sbp_only   --outer 10 --inner 10   --h-grid 1 2 4 8 12 16 20 24   --alpha-grid 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10 100   --fit-final --save-predictions
```

---

## What the script does

1. **Downloads & preprocesses** the South African Heart dataset.
2. Builds a design matrix (`get_dummies`, drop `chd` + target variable).
3. Runs **nested CV** to select the best:
   - `h` = number of hidden units  
   - `α` = L2 regularization 
4. Compares ANN test MSE to a **baseline predictor** (mean of `y_train`).
5. Saves:
   - `*_nested_cv_ann.csv` → outer fold results  
   - `performance_by_h_alpha.csv` → all inner-CV combinations  
   - `performance_by_h_alpha.png` → MSE vs α for each h  
   - Final trained model (`model.joblib`) + metadata (`feature_names.json`, `targets.json`)  

---

## Output files

All artefacts are written to `ann_regression/results/<setting>/`:

| File | Description |
|------|--------------|
| `model.joblib` | Trained ANN model |
| `feature_names.json` | Ordered list of training features |
| `targets.json` | Target variable(s) |
| `predictions.csv` | Actual vs predicted values (training data) |
| `performance_by_h_alpha.csv` | Inner-CV grid results |
| `performance_by_h_alpha.png` | Plot of performance vs α for each h |
| `*_nested_cv_ann.csv` | Outer-fold MSE table (for report) |

---

## Inference on new data

Use `inference.py` to run predictions on any new CSV file:

```bash
python -m ann_regression.inference   --setting ldl_only   --input path/to/new_samples.csv   --output path/to/predictions.csv
```

The script automatically:
- aligns categorical features (`famhist`)
- ensures consistent feature ordering
- and writes a CSV with predicted targets appended.

If the input CSV already includes the true target column, it also prints the **MSE** for validation.

---

## Notes

- The default dataset is loaded from  
  [`https://www.hastie.su.domains/Datasets/SAheart.data`](https://www.hastie.su.domains/Datasets/SAheart.data)
- You can change the target setup using `--setting`:
  - `ldl_only`
  - `sbp_only`
  - `joint_ldl_sbp`
- All results are reproducible (`random_state=0`).

