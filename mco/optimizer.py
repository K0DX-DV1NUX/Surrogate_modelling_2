import csv
import itertools
import math
import time
from dataclasses import asdict

import numpy as np

from simulations import FormationCandidate


class CandidateGrid:
    def __init__(self, grid_config):
        self.grid = grid_config

    def candidates(self):
        keys = list(self.grid)
        for values in itertools.product(*(self.grid[key] for key in keys)):
            yield FormationCandidate(**dict(zip(keys, values)))


class CandidateScorer:
    def __init__(self, weights):
        self.weights = weights

    def rank(self, results):
        objective_values = {
            "formation_time_h": [row["formation_time_h"] for row in results],
            "final_capacity": [row["capacity_final"] for row in results],
            "capacity_fade": [row["capacity_fade"] for row in results],
            "plating_growth": [row["plating_growth"] for row in results],
            "sei_growth": [row["sei_growth"] for row in results],
            "resistance_growth": [row["resistance_growth"] for row in results],
        }
        for result in results:
            penalties = {
                "formation_time_h": self._lower(
                    result["formation_time_h"], objective_values["formation_time_h"]
                ),
                "final_capacity": self._higher(
                    result["capacity_final"], objective_values["final_capacity"]
                ),
                "capacity_fade": self._lower(
                    result["capacity_fade"], objective_values["capacity_fade"]
                ),
                "plating_growth": self._lower(
                    result["plating_growth"], objective_values["plating_growth"]
                ),
                "sei_growth": self._lower(
                    result["sei_growth"], objective_values["sei_growth"]
                ),
                "resistance_growth": self._lower(
                    result["resistance_growth"], objective_values["resistance_growth"]
                ),
            }
            result["score"] = sum(
                self.weights[key] * penalties[key]
                for key in self.weights
            )
        return sorted(results, key=lambda row: row["score"])

    def _lower(self, value, values):
        minimum, maximum = min(values), max(values)
        return 0.0 if math.isclose(minimum, maximum) else (value - minimum) / (maximum - minimum)

    def _higher(self, value, values):
        minimum, maximum = min(values), max(values)
        return 0.0 if math.isclose(minimum, maximum) else (maximum - value) / (maximum - minimum)


class ResultRepository:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def save(self, filename, rows):
        if not rows:
            return
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class FormationOptimizer:
    def __init__(
        self,
        mco_config,
        experiment_factory,
        simulator,
        analyzer,
        results_dir,
        surrogate_predictor=None,
        trajectory_featurizer=None,
    ):
        self.config = mco_config.data
        self.experiments = experiment_factory
        self.simulator = simulator
        self.analyzer = analyzer
        self.grid = CandidateGrid(self.config["grid"])
        self.scorer = CandidateScorer(self.config["weights"])
        self.repository = ResultRepository(results_dir)
        self.surrogate_predictor = surrogate_predictor
        self.trajectory_featurizer = trajectory_featurizer

    def run(self):
        recommendation = self.config["recommendation"]
        candidates = list(self.grid.candidates())
        use_surrogate = self._use_surrogate(recommendation["mode"])
        mode = "surrogate" if use_surrogate else "PyBaMM ageing"
        print(f"Evaluating {len(candidates)} candidates using {mode}")

        pre_step = self.simulator.solve_timed("pre-step", self.experiments.pre_step())
        results = []
        failures = []
        for index, candidate in enumerate(candidates, start=1):
            print(f"Candidate {index}/{len(candidates)}: {candidate}")
            started = time.perf_counter()
            try:
                result = self._evaluate(
                    candidate,
                    pre_step.last_state,
                    recommendation,
                    use_surrogate,
                )
                result["runtime_s"] = time.perf_counter() - started
                results.append(result)
                print(
                    f"  formation={result['formation_time_h']:.2f} h, "
                    f"final capacity={result['capacity_final']:.3f} A.h"
                )
            except Exception as error:
                failures.append({**asdict(candidate), "error": str(error)})
                print(f"  failed: {error}")

        if not results:
            self.repository.save("failed_candidates.csv", failures)
            raise RuntimeError("All recommendation candidates failed")

        ranked = self.scorer.rank(results)
        top_n = recommendation["top_n"]
        self.repository.save("all_candidates.csv", ranked)
        self.repository.save("top_candidates.csv", ranked[:top_n])
        self.repository.save("failed_candidates.csv", failures)
        self._print_top(ranked[:top_n])
        return ranked[:top_n]

    def _use_surrogate(self, mode):
        available = self.surrogate_predictor is not None and self.surrogate_predictor.available()
        if mode == "surrogate" and not available:
            raise FileNotFoundError("Surrogate mode requested but trained models are unavailable")
        if mode == "pybamm":
            return False
        return available

    def _evaluate(self, candidate, pre_step_state, recommendation, use_surrogate):
        formation_solution = self.simulator.solve(
            self.experiments.formation(candidate),
            pre_step_state,
        )
        result = asdict(candidate)
        result["formation_time_h"] = (
            formation_solution.t[-1] - formation_solution.t[0]
        ) / 3600

        if use_surrogate:
            features = self.trajectory_featurizer.extract(formation_solution)
            predictions = self.surrogate_predictor.predict(
                features,
                recommendation["prediction_cycles"],
            )
            metrics = self._metrics_from_predictions(predictions)
        else:
            aging_solution = self.simulator.solve(
                self.experiments.aging(recommendation["aging_cycles"]),
                formation_solution.last_state,
            )
            metrics = self._metrics_from_solution(aging_solution)

        result.update(metrics)
        return result

    def _metrics_from_solution(self, solution):
        return self._metrics_from_pairs(self.analyzer.first_and_last_metrics(solution))

    def _metrics_from_predictions(self, predictions):
        pairs = {
            key: (float(predictions[key].iloc[0]), float(predictions[key].iloc[-1]))
            for key in ("capacity", "plating", "sei", "resistance")
        }
        return self._metrics_from_pairs(pairs)

    def _metrics_from_pairs(self, pairs):
        result = {}
        for key in ("capacity", "plating", "sei", "resistance"):
            start, final = pairs[key]
            result[f"{key}_start"] = start
            result[f"{key}_final"] = final
            result[f"{key}_delta"] = final - start
            result[f"{key}_growth" if key != "capacity" else "capacity_fade"] = abs(final - start)
        return result

    def _print_top(self, rows):
        print("\nRecommended formation protocols")
        print(
            f"{'Rank':>4} {'Score':>8} {'Charge C':>9} {'Discharge C':>12} "
            f"{'Rest':>8} {'Cycles':>7} {'Form h':>9} {'Final cap':>10}"
        )
        print("-" * 86)
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank:>4} {row['score']:>8.3f} {row['x1_charge_rate']:>9.3g} "
                f"{row['x2_discharge_rate']:>12.3g} {row['x3_rest_minutes']:>8.1f} "
                f"{row['x4_num_cycles']:>7} {row['formation_time_h']:>9.2f} "
                f"{row['capacity_final']:>10.3f}"
            )
