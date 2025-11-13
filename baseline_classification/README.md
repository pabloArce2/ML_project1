# Baseline Classification

This module implements a **baseline classifier** for the South African Heart Disease dataset.  
It always predicts the **most frequent class** (`chd`) observed in the training set.  
The purpose of this model is to provide a **reference performance** for classification models,  
such as logistic regression or neural networks.

---

## Description

- **Target:** `chd` (presence of coronary heart disease)
- **Baseline approach:** Majority class prediction
- **Evaluation metric:** Accuracy (mean ± standard deviation)
- **Cross-validation:** 10-fold by default
- **Dataset:** [SAheart dataset](https://www.hastie.su.domains/Datasets/SAheart.data)

---

## Files

| File | Description |
|------|--------------|
| `run.py` | Runs k-fold cross-validation for the majority-class baseline. Saves accuracy metrics, plots, and model artefacts. |
| `inference.py` | Loads the saved baseline model and predicts `chd` for new input data. |
| `results/` | Contains all saved artefacts (model.joblib, CV metrics, predictions, plots, etc.). |

---

## Usage

### Train and evaluate baseline
```bash
python -m baseline_classification.run   --folds 10 --save-predictions
```

This will:
- Compute 10-fold cross-validation accuracy
- Save the majority class and its frequency
- Produce a plot showing accuracy across folds
- Output results under:
  ```
  baseline_classification/results/majority_baseline_classification/
  ```

---

### Inference on new data
Once the model is trained, you can run inference on new CSV files:

```bash
python -m baseline_classification.inference   --input path/to/new_data.csv   --output baseline_classification/results/majority_baseline_classification/preds_on_new_data.csv
```

If the CSV contains the true `chd` column, the script will also print the accuracy.

---

## Output artefacts

| File | Description |
|------|--------------|
| `model.joblib` | Contains the stored majority class and its training-set frequency. |
| `targets.json` | Lists the predicted variable (`chd`). |
| `cv_metrics.csv` | Accuracy per fold. |
| `baseline_classification_fold_acc.png` | Accuracy plot across folds. |
| `predictions.csv` | In-sample predictions (optional). |
| `summary.json` | Summary report with mean and std accuracy. |

---

## Notes

- This baseline provides a simple check:  
  any real classifier should outperform it.

---

## Example Output

```
=== Majority-Class Baseline (Classification) ===
Global majority class for 'chd': 0 (fraction=0.65)
10-Fold CV Accuracy: 0.6521 ± 0.0304
Saved CV metrics to baseline_classification/results/majority_baseline_classification/cv_metrics.csv
```

