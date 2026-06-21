import csv
import itertools
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from simulations import FormationCandidate


class SurrogateDatasetBuilder:
    def __init__(
        self,
        surrogate_config,
        project_config,
        experiment_factory,
        simulator,
        analyzer,
        featurizer,
    ):
        self.config = surrogate_config.data
        self.project_config = project_config
        self.experiments = experiment_factory
        self.simulator = simulator
        self.analyzer = analyzer
        self.featurizer = featurizer
        self.dataset_path = project_config.resolve_path(self.config["paths"]["dataset"])
        self.failed_path = project_config.resolve_path(
            self.config["paths"]["failed_candidates"]
        )

    def create(self, force=False, limit_candidates=None):
        if self.dataset_path.exists() and not force:
            print(f"Dataset already exists; skipping creation: {self.dataset_path}")
            return self.dataset_path

        candidates = list(self._candidates())
        if limit_candidates is not None:
            candidates = candidates[:limit_candidates]

        pre_step = self.simulator.solve_timed("pre-step", self.experiments.pre_step())
        aging_cycles = self.config["data"]["aging_cycles"]
        mesh = self.config["data"]["training_ageing_mesh"]
        rows = []
        failures = []

        for index, candidate in enumerate(candidates, start=1):
            print(f"Dataset candidate {index}/{len(candidates)}: {candidate}")
            started = time.perf_counter()
            try:
                formation = self.simulator.solve(
                    self.experiments.formation(candidate),
                    pre_step.last_state,
                )
                aging = self.simulator.solve(
                    self.experiments.aging(aging_cycles),
                    formation.last_state,
                )
                features = self.featurizer.extract(formation)
                series = {
                    key: self.analyzer.metric_series(aging, key)
                    for key in ("capacity", "plating", "sei", "resistance")
                }
                for cycle_number in mesh:
                    index_in_series = cycle_number - 1
                    if index_in_series < 0 or any(
                        index_in_series >= len(values)
                        for values in series.values()
                    ):
                        continue
                    targets = {
                        key: values[index_in_series]
                        for key, values in series.items()
                    }
                    if not all(np.isfinite(value) for value in targets.values()):
                        continue
                    rows.append(
                        {
                            "protocol_key": self._protocol_key(candidate),
                            **asdict(candidate),
                            **features,
                            "ageing_cycle": cycle_number,
                            **targets,
                        }
                    )
                print(f"  completed in {time.perf_counter() - started:.2f} s")
            except Exception as error:
                failures.append({**asdict(candidate), "error": str(error)})
                print(f"  failed: {error}")

        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(self.dataset_path, index=False)
        self._write_failures(failures)
        print(f"Saved {len(rows)} rows to {self.dataset_path}")
        return self.dataset_path

    def _candidates(self):
        grid = self.config["grid"]
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            yield FormationCandidate(**dict(zip(keys, values)))

    def _protocol_key(self, candidate):
        return (
            f"x1={candidate.x1_charge_rate}|x2={candidate.x2_discharge_rate}|"
            f"x3={candidate.x3_rest_minutes}|x4={candidate.x4_num_cycles}"
        )

    def _write_failures(self, rows):
        if not rows:
            return
        self.failed_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failed_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
