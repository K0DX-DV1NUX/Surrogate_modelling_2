import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import (
    build_aging_experiment,
    build_formation_experiment,
    build_pre_step_experiment,
)
from mco import OUTPUT_VARIABLES
from model import run_model
from outputs import cycle_variable_max
from parameters import load_parameter_values


DATA_DIR = Path("surrogate_data")
MODELS_DIR = Path("surrogate_models")
PLOTS_DIR = Path("surrogate_plots")

SURROGATE_GRID = {
    "x1_charge_rate": [0.01, 0.5],
    "x2_discharge_rate": [0.01, 0.5],
    "x3_rest_minutes": [0.1,],
    "x4_num_cycles": [1, 3],
}

DEFAULT_SELECTED_CYCLES = [0, 10, 30, 50, 70, 100]

TARGET_COLUMNS = [
    "capacity",
    "plating",
    "sei",
    "resistance",
]

FEATURE_COLUMNS = [
    "x1_charge_rate",
    "x2_discharge_rate",
    "x3_rest_minutes",
    "x4_num_cycles",
    "formation_time_h",
    "post_formation_capacity",
    "post_formation_plating",
    "post_formation_sei",
    "post_formation_resistance",
    "ageing_cycle",
]


@dataclass(frozen=True)
class FormationCandidate:
    x1_charge_rate: float
    x2_discharge_rate: float
    x3_rest_minutes: int
    x4_num_cycles: int


def parse_cycle_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_candidates(grid=SURROGATE_GRID):
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield FormationCandidate(**dict(zip(keys, values)))


def protocol_key(candidate):
    return (
        f"x1={candidate.x1_charge_rate}|"
        f"x2={candidate.x2_discharge_rate}|"
        f"x3={candidate.x3_rest_minutes}|"
        f"x4={candidate.x4_num_cycles}"
    )


def finite_cycle_values(solution, variable_name):
    values = [
        cycle_variable_max(cycle, variable_name)
        for cycle in solution.cycles
    ]
    return [value for value in values if np.isfinite(value)]


def value_at_ageing_cycle(solution, variable_name, ageing_cycle):
    values = finite_cycle_values(solution, variable_name)
    if not values:
        return np.nan
    if ageing_cycle >= len(values):
        return np.nan
    return values[ageing_cycle]


def last_finite_value(solution, variable_name):
    values = finite_cycle_values(solution, variable_name)
    if not values:
        return np.nan
    return values[-1]


def formation_features(candidate, formation_solution):
    return {
        **asdict(candidate),
        "formation_time_h": (formation_solution.t[-1] - formation_solution.t[0]) / 3600,
        "post_formation_capacity": last_finite_value(
            formation_solution, OUTPUT_VARIABLES["capacity"]
        ),
        "post_formation_plating": last_finite_value(
            formation_solution, OUTPUT_VARIABLES["plating"]
        ),
        "post_formation_sei": last_finite_value(
            formation_solution, OUTPUT_VARIABLES["sei"]
        ),
        "post_formation_resistance": last_finite_value(
            formation_solution, OUTPUT_VARIABLES["resistance"]
        ),
    }


def ageing_targets(aging_solution, ageing_cycle):
    return {
        "capacity": value_at_ageing_cycle(
            aging_solution, OUTPUT_VARIABLES["capacity"], ageing_cycle
        ),
        "plating": value_at_ageing_cycle(
            aging_solution, OUTPUT_VARIABLES["plating"], ageing_cycle
        ),
        "sei": value_at_ageing_cycle(
            aging_solution, OUTPUT_VARIABLES["sei"], ageing_cycle
        ),
        "resistance": value_at_ageing_cycle(
            aging_solution, OUTPUT_VARIABLES["resistance"], ageing_cycle
        ),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(
    data_path,
    max_ageing_cycles,
    selected_cycles,
    limit_candidates=None,
):
    parameter_values = load_parameter_values()
    candidates = list(build_candidates())
    if limit_candidates is not None:
        candidates = candidates[:limit_candidates]

    print(f"Running pre-step once for {len(candidates)} surrogate candidates")
    pre_step_solution = run_model(
        build_pre_step_experiment(),
        parameter_values,
    )
    aging_experiment = build_aging_experiment(max_ageing_cycles)

    rows = []
    failures = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"\nSurrogate data candidate {index}/{len(candidates)}: {candidate}")
        start_time = time.time()
        try:
            formation_solution = run_model(
                build_formation_experiment(**asdict(candidate)),
                parameter_values,
                pre_step_solution.last_state,
            )
            aging_solution = run_model(
                aging_experiment,
                parameter_values,
                formation_solution.last_state,
            )
        except Exception as error:
            failures.append({**asdict(candidate), "error": str(error)})
            print(f"Failed: {error}")
            continue

        base_features = formation_features(candidate, formation_solution)
        for ageing_cycle in selected_cycles:
            targets = ageing_targets(aging_solution, ageing_cycle)
            if any(not np.isfinite(value) for value in targets.values()):
                continue

            rows.append({
                "protocol_key": protocol_key(candidate),
                **base_features,
                "ageing_cycle": ageing_cycle,
                **targets,
            })

        print(f"Completed in {time.time() - start_time:.2f}s")

    write_csv(data_path, rows)
    write_csv(data_path.parent / "failed_surrogate_candidates.csv", failures)
    print(f"\nSaved surrogate dataset: {data_path}")
    print(f"Rows: {len(rows)}")
    if failures:
        print(f"Failures: {len(failures)}")


def split_by_protocol(data):
    groups = data["protocol_key"]
    unique_groups = groups.nunique()
    if unique_groups < 3:
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)
        return train_data, val_data, test_data

    train_val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(
            data, groups=groups
        )
    )
    train_val = data.iloc[train_val_idx]
    test_data = data.iloc[test_idx]

    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=43).split(
            train_val, groups=train_val["protocol_key"]
        )
    )
    return train_val.iloc[train_idx], train_val.iloc[val_idx], test_data


def build_model(random_state=42):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calculate_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
    }


def save_prediction_plot(path, target_name, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions["actual"], predictions["predicted"], alpha=0.75)
    minimum = min(predictions["actual"].min(), predictions["predicted"].min())
    maximum = max(predictions["actual"].max(), predictions["predicted"].max())
    plt.plot([minimum, maximum], [minimum, maximum], color="black", linewidth=1)
    plt.xlabel("PyBaMM")
    plt.ylabel("Surrogate")
    plt.title(target_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def train_surrogates(data_path, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR):
    data = pd.read_csv(data_path)
    data = data.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    if data.empty:
        raise ValueError(f"No valid training rows found in {data_path}")

    train_data, val_data, test_data = split_by_protocol(data)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    prediction_rows = []
    model_files = {}

    for target_name in TARGET_COLUMNS:
        model = build_model()
        model.fit(train_data[FEATURE_COLUMNS], train_data[target_name])
        model_file = f"{target_name}_random_forest.joblib"
        joblib.dump(model, models_dir / model_file)
        model_files[target_name] = model_file

        for split_name, split_data in (
            ("train", train_data),
            ("validation", val_data),
            ("test", test_data),
        ):
            predicted = model.predict(split_data[FEATURE_COLUMNS])
            metrics = calculate_metrics(split_data[target_name], predicted)
            metrics_rows.append({
                "target": target_name,
                "split": split_name,
                "rows": len(split_data),
                **metrics,
            })

            for actual, pred, protocol, cycle in zip(
                split_data[target_name],
                predicted,
                split_data["protocol_key"],
                split_data["ageing_cycle"],
            ):
                prediction_rows.append({
                    "target": target_name,
                    "split": split_name,
                    "protocol_key": protocol,
                    "ageing_cycle": cycle,
                    "actual": actual,
                    "predicted": pred,
                    "error": pred - actual,
                })

        test_predictions = pd.DataFrame([
            {
                "actual": actual,
                "predicted": pred,
            }
            for actual, pred in zip(
                test_data[target_name],
                model.predict(test_data[FEATURE_COLUMNS]),
            )
        ])
        save_prediction_plot(
            plots_dir / f"{target_name}_test_prediction.png",
            target_name,
            test_predictions,
        )

    pd.DataFrame(metrics_rows).to_csv(models_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(models_dir / "predictions.csv", index=False)
    metadata = {
        "model_type": "RandomForestRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "model_files": model_files,
        "data_path": str(data_path),
        "rows": len(data),
        "train_rows": len(train_data),
        "validation_rows": len(val_data),
        "test_rows": len(test_data),
    }
    with (models_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved surrogate models in {models_dir}")
    print(pd.DataFrame(metrics_rows).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate data and train RandomForest ageing surrogate models."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "ageing_surrogate_dataset.csv",
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--max-ageing-cycles", type=int, default=100)
    parser.add_argument(
        "--selected-cycles",
        type=parse_cycle_list,
        default=DEFAULT_SELECTED_CYCLES,
        help="Comma-separated ageing cycles, e.g. 0,100,250,500,1000",
    )
    parser.add_argument(
        "--limit-candidates",
        type=int,
        default=None,
        help="Limit candidates for a quick smoke run.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Train using an existing CSV dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_generate:
        generate_dataset(
            data_path=args.data_path,
            max_ageing_cycles=args.max_ageing_cycles,
            selected_cycles=args.selected_cycles,
            limit_candidates=args.limit_candidates,
        )

    train_surrogates(
        data_path=args.data_path,
        models_dir=args.models_dir,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
