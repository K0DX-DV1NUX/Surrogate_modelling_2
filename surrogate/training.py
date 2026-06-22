import json
import math

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SurrogateTrainer:
    TARGETS = ("capacity", "plating", "sei", "resistance")

    def __init__(self, surrogate_config, project_config, featurizer):
        self.config = surrogate_config.data
        self.project_config = project_config
        self.featurizer = featurizer
        paths = self.config["paths"]
        self.dataset_path = project_config.resolve_path(paths["dataset"])
        self.models_dir = project_config.resolve_path(paths["models_dir"])
        self.plots_dir = project_config.resolve_path(paths["plots_dir"])

    def train(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}. Run surrogate/create_dataset.py first."
            )
        data = pd.read_csv(self.dataset_path)
        features = self.featurizer.feature_columns()
        data = data.dropna(subset=features + list(self.TARGETS))
        train, validation, test = self._split(data)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        metrics = []
        predictions = []
        model_files = {}
        for target in self.TARGETS:
            model = self._model()
            model.fit(train[features], train[target])
            model_file = f"{target}_xgboost.joblib"
            joblib.dump(model, self.models_dir / model_file)
            model_files[target] = model_file

            for split_name, split_data in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            ):
                predicted = model.predict(split_data[features])
                metrics.append(
                    {
                        "target": target,
                        "split": split_name,
                        "rows": len(split_data),
                        **self._metrics(split_data[target], predicted),
                    }
                )
                predictions.extend(
                    {
                        "target": target,
                        "split": split_name,
                        "protocol_key": protocol,
                        "ageing_cycle": cycle,
                        "actual": actual,
                        "predicted": prediction,
                        "error": prediction - actual,
                    }
                    for actual, prediction, protocol, cycle in zip(
                        split_data[target],
                        predicted,
                        split_data["protocol_key"],
                        split_data["ageing_cycle"],
                    )
                )

        metrics_frame = pd.DataFrame(metrics)
        predictions_frame = pd.DataFrame(predictions)
        metrics_frame.to_csv(self.models_dir / "metrics.csv", index=False)
        predictions_frame.to_csv(self.models_dir / "predictions.csv", index=False)
        self._plot_trajectories(predictions_frame)
        self._save_metadata(model_files, features, data, train, validation, test)
        print(metrics_frame.to_string(index=False))

    def _model(self):
        try:
            from xgboost import XGBRegressor
        except Exception as error:
            raise RuntimeError(
                "XGBoost could not load. On macOS, install the OpenMP runtime "
                "(libomp.dylib) before training."
            ) from error

        model = self.config["xgboost"]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", XGBRegressor(**model)),
            ]
        )

    def _split(self, data):
        groups = data["protocol_key"]
        split = self.config["split"]
        if groups.nunique() < 3:
            train, test = train_test_split(
                data,
                test_size=split["test_fraction"],
                random_state=split["test_random_state"],
            )
            train, validation = train_test_split(
                train,
                test_size=split["validation_fraction_of_train"],
                random_state=split["validation_random_state"],
            )
            return train, validation, test

        train_indices, test_indices = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=split["test_fraction"],
                random_state=split["test_random_state"],
            ).split(data, groups=groups)
        )
        train_pool = data.iloc[train_indices]
        test = data.iloc[test_indices]
        train_indices, validation_indices = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=split["validation_fraction_of_train"],
                random_state=split["validation_random_state"],
            ).split(train_pool, groups=train_pool["protocol_key"])
        )
        return train_pool.iloc[train_indices], train_pool.iloc[validation_indices], test

    def _metrics(self, actual, predicted):
        return {
            "mae": mean_absolute_error(actual, predicted),
            "rmse": math.sqrt(mean_squared_error(actual, predicted)),
            "r2": r2_score(actual, predicted) if len(actual) > 1 else np.nan,
        }

    def _plot_trajectories(self, predictions):
        max_protocols = self.config["plots"]["max_test_protocols"]
        for target in self.TARGETS:
            figure, axis = plt.subplots(figsize=(11, 6))
            subset = predictions[
                (predictions["target"] == target)
                & (predictions["split"] == "test")
            ]
            selected_protocols = (
                subset["protocol_key"].drop_duplicates().iloc[:max_protocols]
            )
            subset = subset[subset["protocol_key"].isin(selected_protocols)]

            for color_index, (protocol, values) in enumerate(
                subset.groupby("protocol_key", sort=False)
            ):
                values = values.sort_values("ageing_cycle")
                color = plt.get_cmap("tab10")(color_index)
                axis.plot(
                    values["ageing_cycle"],
                    values["actual"],
                    color=color,
                    linewidth=1.6,
                    label=f"{protocol} - PyBaMM",
                )
                axis.scatter(
                    values["ageing_cycle"],
                    values["actual"],
                    color=color,
                    marker="o",
                    s=28,
                )
                axis.plot(
                    values["ageing_cycle"],
                    values["predicted"],
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    label=f"{protocol} - surrogate",
                )
                axis.scatter(
                    values["ageing_cycle"],
                    values["predicted"],
                    color=color,
                    marker="x",
                    s=38,
                )
            axis.set_xlabel("Ageing cycle")
            axis.set_ylabel(target)
            axis.set_title(f"{target} trajectory")
            axis.grid(True, alpha=0.3)
            axis.legend(
                fontsize=7,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
            )
            figure.tight_layout()
            figure.savefig(self.plots_dir / f"{target}_trajectory_test.png", dpi=300)
            plt.close(figure)

    def _save_metadata(self, model_files, features, data, train, validation, test):
        metadata = {
            "model_type": "XGBRegressor",
            "input_type": "formation_voltage_current_charge_trajectory",
            "trajectory_feature_count": self.featurizer.feature_count,
            "feature_columns": features,
            "target_columns": list(self.TARGETS),
            "model_files": model_files,
            "dataset": str(self.dataset_path),
            "rows": len(data),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        }
        with (self.models_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
