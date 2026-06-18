import json
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODELS_DIR = Path("surrogate_models")


def load_surrogate_models(models_dir=DEFAULT_MODELS_DIR):
    models_path = Path(models_dir)
    with (models_path / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    models = {}
    for target_name, model_file in metadata["model_files"].items():
        models[target_name] = joblib.load(models_path / model_file)

    return {
        "models": models,
        "metadata": metadata,
        "feature_columns": metadata["feature_columns"],
    }


def predict_ageing_points(features, cycle_numbers, models_dir=DEFAULT_MODELS_DIR):
    surrogate = load_surrogate_models(models_dir)
    rows = []

    for cycle_number in cycle_numbers:
        row = dict(features)
        row["ageing_cycle"] = cycle_number
        rows.append(row)

    feature_frame = pd.DataFrame(rows)
    feature_frame = feature_frame.reindex(columns=surrogate["feature_columns"])

    predictions = pd.DataFrame({"ageing_cycle": cycle_numbers})
    for target_name, model in surrogate["models"].items():
        predictions[target_name] = model.predict(feature_frame)

    return predictions
