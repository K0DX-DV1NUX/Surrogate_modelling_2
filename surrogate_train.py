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
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import (
    build_aging_experiment,
    build_formation_experiment,
    build_pre_step_experiment,
)
from mco import OUTPUT_VARIABLES
from model import run_model
from outputs import cycle_metric_value, experiment_cycles
from parameters import load_parameter_values


DATA_DIR = Path("surrogate_data")
MODELS_DIR = Path("surrogate_models")
PLOTS_DIR = Path("surrogate_plots")

SURROGATE_GRID = {
    "x1_charge_rate": [0.01, 0.5],
    "x2_discharge_rate": [0.01, 0.5],
    "x3_rest_minutes": [0.1, 60],
    "x4_num_cycles": [1, 3],
}

TARGET_COLUMNS = [
    "capacity",
    "plating",
    "sei",
    "resistance",
]

FORMATION_VARIABLES = {
    "voltage": "Terminal voltage [V]",
    "current": "Current [A]",
}

DEFAULT_AGEING_CYCLES = 500
DEFAULT_TRAINING_AGEING_MESH = [
    0,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
]
DEFAULT_TRAJECTORY_FEATURES = 256


@dataclass(frozen=True)
class FormationCandidate:
    x1_charge_rate: float
    x2_discharge_rate: float
    x3_rest_minutes: float
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


def as_1d_array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def cumulative_trapezoid(y_values, x_values):
    y_values = as_1d_array(y_values)
    x_values = as_1d_array(x_values)
    if len(y_values) < 2:
        return np.zeros_like(y_values)

    increments = 0.5 * (y_values[1:] + y_values[:-1]) * np.diff(x_values)
    return np.concatenate([[0.0], np.cumsum(increments)])


def solution_variable(solution, variable_name):
    return as_1d_array(solution[variable_name].entries)


def unique_time_series(time_s, *series):
    time_s = as_1d_array(time_s)
    unique_time, unique_indices = np.unique(time_s, return_index=True)
    return [unique_time] + [as_1d_array(values)[unique_indices] for values in series]


def interpolate_uniform_time(time_s, values, feature_count):
    if len(time_s) == 0:
        return np.zeros(feature_count)
    if len(time_s) == 1 or math.isclose(time_s[-1], time_s[0]):
        return np.full(feature_count, values[-1])

    uniform_time = np.linspace(time_s[0], time_s[-1], feature_count)
    return np.interp(uniform_time, time_s, values)


def haar_average_pool(values, output_length):
    values = as_1d_array(values)
    if len(values) == 0:
        return np.zeros(output_length)

    while len(values) >= output_length * 2:
        if len(values) % 2 == 1:
            values = values[:-1]
        values = 0.5 * (values[0::2] + values[1::2])

    source_x = np.linspace(0.0, 1.0, len(values))
    target_x = np.linspace(0.0, 1.0, output_length)
    return np.interp(target_x, source_x, values)


def trajectory_features(formation_solution, feature_count=DEFAULT_TRAJECTORY_FEATURES):
    time_s = solution_variable(formation_solution, "Time [s]")
    voltage = solution_variable(formation_solution, FORMATION_VARIABLES["voltage"])
    current = solution_variable(formation_solution, FORMATION_VARIABLES["current"])
    time_s, voltage, current = unique_time_series(time_s, voltage, current)

    time_h = (time_s - time_s[0]) / 3600
    q_signed_ah = cumulative_trapezoid(current, time_s) / 3600
    q_abs_ah = cumulative_trapezoid(np.abs(current), time_s) / 3600

    raw_signals = {
        "voltage": voltage,
        "current": current,
        "q_signed": q_signed_ah,
        "q_abs": q_abs_ah,
    }

    features = {}
    for signal_name, signal_values in raw_signals.items():
        uniform_signal = interpolate_uniform_time(time_h, signal_values, len(signal_values))
        pooled_signal = haar_average_pool(uniform_signal, feature_count)
        for index, value in enumerate(pooled_signal):
            features[f"{signal_name}_{index:03d}"] = value

    return features


def trajectory_feature_columns(feature_count=DEFAULT_TRAJECTORY_FEATURES):
    columns = []
    for signal_name in ("voltage", "current", "q_signed", "q_abs"):
        columns.extend(
            f"{signal_name}_{index:03d}"
            for index in range(feature_count)
        )
    columns.append("ageing_cycle")
    return columns


def finite_cycle_values(solution, variable_name):
    values = [
        cycle_metric_value(cycle, variable_name)
        for cycle in experiment_cycles(solution)
    ]
    return [value for value in values if np.isfinite(value)]


def value_at_ageing_cycle(solution, variable_name, ageing_cycle):
    values = finite_cycle_values(solution, variable_name)
    if not values or ageing_cycle >= len(values):
        return np.nan
    return values[ageing_cycle]


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
    training_ageing_mesh,
    trajectory_feature_count,
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

        base_features = trajectory_features(
            formation_solution,
            feature_count=trajectory_feature_count,
        )
        for ageing_cycle in training_ageing_mesh:
            targets = ageing_targets(aging_solution, ageing_cycle)
            if any(not np.isfinite(value) for value in targets.values()):
                continue

            rows.append({
                "protocol_key": protocol_key(candidate),
                **asdict(candidate),
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
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
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


def save_trajectory_plots(path, prediction_data):
    path.mkdir(parents=True, exist_ok=True)
    for target_name in TARGET_COLUMNS:
        plt.figure(figsize=(9, 5))
        subset = prediction_data[
            (prediction_data["target"] == target_name)
            & (prediction_data["split"] == "test")
        ]
        for protocol, protocol_data in subset.groupby("protocol_key"):
            protocol_data = protocol_data.sort_values("ageing_cycle")
            plt.plot(
                protocol_data["ageing_cycle"],
                protocol_data["actual"],
                marker="o",
                linewidth=1.5,
                label=f"{protocol} PyBaMM",
            )
            plt.plot(
                protocol_data["ageing_cycle"],
                protocol_data["predicted"],
                linestyle="--",
                linewidth=1.5,
                label=f"{protocol} surrogate",
            )

        plt.xlabel("Ageing cycle")
        plt.ylabel(target_name)
        plt.title(f"{target_name} trajectory")
        plt.grid(True, alpha=0.3)
        if len(subset["protocol_key"].unique()) <= 4:
            plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(path / f"{target_name}_trajectory_test.png", dpi=300)
        plt.close()


def train_surrogates(
    data_path,
    models_dir=MODELS_DIR,
    plots_dir=PLOTS_DIR,
    trajectory_feature_count=DEFAULT_TRAJECTORY_FEATURES,
):
    data = pd.read_csv(data_path)
    feature_columns = trajectory_feature_columns(trajectory_feature_count)
    data = data.dropna(subset=feature_columns + TARGET_COLUMNS)
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
        model.fit(train_data[feature_columns], train_data[target_name])
        model_file = f"{target_name}_random_forest.joblib"
        joblib.dump(model, models_dir / model_file)
        model_files[target_name] = model_file

        for split_name, split_data in (
            ("train", train_data),
            ("validation", val_data),
            ("test", test_data),
        ):
            predicted = model.predict(split_data[feature_columns])
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
                model.predict(test_data[feature_columns]),
            )
        ])
        save_prediction_plot(
            plots_dir / f"{target_name}_test_prediction.png",
            target_name,
            test_predictions,
        )

    prediction_data = pd.DataFrame(prediction_rows)
    pd.DataFrame(metrics_rows).to_csv(models_dir / "metrics.csv", index=False)
    prediction_data.to_csv(models_dir / "predictions.csv", index=False)
    save_trajectory_plots(plots_dir, prediction_data)

    metadata = {
        "model_type": "RandomForestRegressor",
        "input_type": "formation_voltage_current_charge_trajectory",
        "trajectory_feature_count": trajectory_feature_count,
        "feature_columns": feature_columns,
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
        description="Generate data and train trajectory-based RandomForest ageing surrogates."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "ageing_trajectory_surrogate_dataset.csv",
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--max-ageing-cycles", type=int, default=DEFAULT_AGEING_CYCLES)
    parser.add_argument(
        "--training-ageing-mesh",
        type=parse_cycle_list,
        default=DEFAULT_TRAINING_AGEING_MESH,
        help="Comma-separated ageing cycles, e.g. 0,1,2,5,10,20,50,100,500",
    )
    parser.add_argument(
        "--trajectory-feature-count",
        type=int,
        default=DEFAULT_TRAJECTORY_FEATURES,
        help="Fixed number of Haar-pooled points per formation signal.",
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
            training_ageing_mesh=args.training_ageing_mesh,
            trajectory_feature_count=args.trajectory_feature_count,
            limit_candidates=args.limit_candidates,
        )

    train_surrogates(
        data_path=args.data_path,
        models_dir=args.models_dir,
        plots_dir=args.plots_dir,
        trajectory_feature_count=args.trajectory_feature_count,
    )


if __name__ == "__main__":
    main()
