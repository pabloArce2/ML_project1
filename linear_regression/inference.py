"""
Simple playground script for generating predictions with the fitted ridge models.

Edit `TARGET_SETTING`, `CUSTOM_ROWS`, or `OUTPUT_PATH` below to experiment with new
inputs. Before running this script, execute `python -m linear_regression.run` so the
trained model artefacts (model.joblib, feature_names.json) exist inside
`linear_regression/results/<target_setting>/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from joblib import load

from .run import RESULTS_DIR

# Choose which trained target configuration to use.
# Options (after running the trainer): "ldl_only", "sbp_only", "joint_ldl_sbp"
TARGET_SETTING = "sbp_only"

# Modify or extend this list with new samples you want to score.
# Make sure to provide all feature columns used during training; any missing columns
# will be filled with zero. Example values shown here are placeholders.
CUSTOM_ROWS: List[Dict[str, Any]] = [
    {
        "tobacco": 5.5,
        "ldl": 4.0,
        "adiposity": 25.0,
        "typea": 45.0,
        "obesity": 26.0,
        "alcohol": 10.0,
        "age": 55.0,
        "famhist_Present": 1.0,  # 1 -> "Present", 0 -> "Absent"
    }
]

# Where to store the scored rows (including predictions). Set to None to skip saving.
OUTPUT_PATH: Path | None = RESULTS_DIR / TARGET_SETTING / "manual_inference.csv"


def load_artifacts(setting: str) -> Tuple[pd.Index, List[str], Any]:
    """Load the saved pipeline and feature ordering for a given target setting."""
    setting_dir = RESULTS_DIR / setting
    model_path = setting_dir / "model.joblib"
    feature_names_path = setting_dir / "feature_names.json"
    targets_path = setting_dir / "targets.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model for '{setting}'. Run `python -m linear_regression.run --setting {setting}` first."
        )
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"Missing feature name list for '{setting}'. Did the training complete successfully?"
        )
    if not targets_path.exists():
        raise FileNotFoundError(
            f"Missing target metadata for '{setting}'. Run the training script again to regenerate artefacts."
        )

    model = load(model_path)
    with feature_names_path.open("r", encoding="utf-8") as handle:
        feature_names = pd.Index(json.load(handle))
    with targets_path.open("r", encoding="utf-8") as handle:
        targets = json.load(handle)
    return feature_names, targets, model


def prepare_features(feature_names: pd.Index, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame aligned with the stored feature order, filling missing columns with zero."""
    df = pd.DataFrame(rows)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df.loc[:, feature_names]


def main() -> None:
    feature_names, targets, model = load_artifacts(TARGET_SETTING)
    ordered_inputs = prepare_features(feature_names, CUSTOM_ROWS)

    preds = model.predict(ordered_inputs.values)
    if preds.ndim == 1:
        preds_df = pd.DataFrame(preds, columns=targets)
    else:
        preds_df = pd.DataFrame(preds, columns=targets)

    # Combine original (unordered) inputs with predictions for readability.
    original_df = pd.DataFrame(CUSTOM_ROWS)
    output_df = pd.concat([original_df.reset_index(drop=True), preds_df], axis=1)

    print(f"\nPredictions for setting '{TARGET_SETTING}':")
    print(output_df)

    if OUTPUT_PATH is not None:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved manual inference results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
