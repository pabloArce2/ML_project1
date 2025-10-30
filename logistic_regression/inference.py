"""
Utility script for interactive scoring with the fitted classification models.

Before running this file, ensure `python -m logistic_regression.run --model logistic`
has been executed so the trained artefacts exist under
`logistic_regression/results/logistic_regression/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load

from .run import RESULTS_DIR, TARGET_NAME

# Choose the trained configuration you wish to load (e.g. "logistic_regression" or "knn_classifier").
MODEL_NAME = "logistic_regression"

# Define samples to score. Provide the same columns used during training; missing columns will be filled with zero.
CUSTOM_ROWS: List[Dict[str, Any]] = [
    {
        "tobacco": 3.0,
        "ldl": 4.2,
        "adiposity": 25.0,
        "typea": 50.0,
        "obesity": 27.5,
        "alcohol": 5.0,
        "age": 52,
        "sbp": 140,
        "famhist_Present": 1.0,
    }
]

# Optional output path for the predictions.
OUTPUT_PATH: Path | None = RESULTS_DIR / MODEL_NAME / "manual_inference.csv"


def load_artifacts(model_name: str) -> Tuple[pd.Index, Any]:
    result_dir = RESULTS_DIR / model_name
    model_path = result_dir / "model.joblib"
    feature_names_path = result_dir / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model for '{model_name}'. Run `python -m logistic_regression.run --model {model_name}` first."
        )
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"Missing feature name list for '{model_name}'. Ensure training completed successfully."
        )

    model = load(model_path)
    with feature_names_path.open("r", encoding="utf-8") as handle:
        feature_names = pd.Index(json.load(handle))
    return feature_names, model


def prepare_features(feature_names: pd.Index, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df.loc[:, feature_names]


def main() -> None:
    feature_names, model = load_artifacts(MODEL_NAME)
    ordered_inputs = prepare_features(feature_names, CUSTOM_ROWS)

    proba = model.predict_proba(ordered_inputs.values)[:, 1]
    proba = np.clip(proba, 1e-9, 1 - 1e-9)
    preds = (proba >= 0.5).astype(int)
    preds_df = pd.DataFrame(
        {
            f"{TARGET_NAME}_probability": proba,
            f"{TARGET_NAME}_prediction": preds,
        }
    )

    original_df = pd.DataFrame(CUSTOM_ROWS)
    output_df = pd.concat([original_df.reset_index(drop=True), preds_df], axis=1)

    print(f"\nPredictions for model '{MODEL_NAME}':")
    print(output_df)

    if OUTPUT_PATH is not None:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved manual inference results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
