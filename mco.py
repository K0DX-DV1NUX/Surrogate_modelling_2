import csv
import itertools
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from experiments import (
    build_aging_experiment,
    build_formation_experiment,
    build_pre_step_experiment,
)
from model import run_model
from outputs import cycle_metric_value, experiment_cycles
from parameters import load_parameter_values


AGEING_CYCLES = 100
TOP_N = 4
RESULTS_DIR = Path("mco_results")

GRID = {
    "x1_charge_rate": [0.01],
    "x2_discharge_rate": [0.01],
    "x3_rest_minutes": [1],
    "x4_num_cycles": [1, 3],
}

OBJECTIVE_WEIGHTS = {
    "formation_time_h": 0.20,
    "final_capacity": 0.20,
    "capacity_fade": 0.15,
    "plating_growth": 0.20,
    "sei_growth": 0.20,
    "resistance_growth": 0.05,
}

OUTPUT_VARIABLES = {
    "capacity": "Discharge capacity [A.h]",
    "plating": "Loss of capacity to negative lithium plating [A.h]",
    "sei": "Loss of capacity to negative SEI [A.h]",
    "resistance": "Local ECM resistance [Ohm]",
}


@dataclass(frozen=True)
class FormationCandidate:
    x1_charge_rate: float
    x2_discharge_rate: float
    x3_rest_minutes: int
    x4_num_cycles: int


def build_candidates(grid=GRID):
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield FormationCandidate(**dict(zip(keys, values)))


def finite_cycle_values(solution, variable_name):
    values = [
        cycle_metric_value(cycle, variable_name)
        for cycle in experiment_cycles(solution)
    ]
    return [value for value in values if np.isfinite(value)]


def first_and_last(solution, variable_name):
    values = finite_cycle_values(solution, variable_name)
    if not values:
        raise ValueError(f"No finite values found for '{variable_name}'")
    return values[0], values[-1]


def evaluate_ageing_solution(solution):
    capacity_start, capacity_final = first_and_last(
        solution, OUTPUT_VARIABLES["capacity"]
    )
    plating_start, plating_final = first_and_last(
        solution, OUTPUT_VARIABLES["plating"]
    )
    sei_start, sei_final = first_and_last(
        solution, OUTPUT_VARIABLES["sei"]
    )
    resistance_start, resistance_final = first_and_last(
        solution, OUTPUT_VARIABLES["resistance"]
    )

    return {
        "capacity_start": capacity_start,
        "capacity_final": capacity_final,
        "capacity_delta": capacity_final - capacity_start,
        "capacity_fade": abs(capacity_final - capacity_start),
        "plating_start": plating_start,
        "plating_final": plating_final,
        "plating_delta": plating_final - plating_start,
        "plating_growth": abs(plating_final - plating_start),
        "sei_start": sei_start,
        "sei_final": sei_final,
        "sei_delta": sei_final - sei_start,
        "sei_growth": abs(sei_final - sei_start),
        "resistance_start": resistance_start,
        "resistance_final": resistance_final,
        "resistance_delta": resistance_final - resistance_start,
        "resistance_growth": abs(resistance_final - resistance_start),
    }


def normalize_lower_is_better(value, values):
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum):
        return 0.0
    return (value - minimum) / (maximum - minimum)


def normalize_higher_is_better(value, values):
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum):
        return 0.0
    return (maximum - value) / (maximum - minimum)


def add_scores(results):
    objective_values = {
        "formation_time_h": [result["formation_time_h"] for result in results],
        "final_capacity": [result["capacity_final"] for result in results],
        "capacity_fade": [result["capacity_fade"] for result in results],
        "plating_growth": [result["plating_growth"] for result in results],
        "sei_growth": [result["sei_growth"] for result in results],
        "resistance_growth": [result["resistance_growth"] for result in results],
    }

    for result in results:
        normalized = {
            "formation_time_h": normalize_lower_is_better(
                result["formation_time_h"], objective_values["formation_time_h"]
            ),
            "final_capacity": normalize_higher_is_better(
                result["capacity_final"], objective_values["final_capacity"]
            ),
            "capacity_fade": normalize_lower_is_better(
                result["capacity_fade"], objective_values["capacity_fade"]
            ),
            "plating_growth": normalize_lower_is_better(
                result["plating_growth"], objective_values["plating_growth"]
            ),
            "sei_growth": normalize_lower_is_better(
                result["sei_growth"], objective_values["sei_growth"]
            ),
            "resistance_growth": normalize_lower_is_better(
                result["resistance_growth"], objective_values["resistance_growth"]
            ),
        }
        result["score"] = sum(
            OBJECTIVE_WEIGHTS[key] * normalized[key]
            for key in OBJECTIVE_WEIGHTS
        )

    return sorted(results, key=lambda item: item["score"])


def run_candidate(candidate, parameter_values, pre_step_last_state, aging_experiment):
    formation_experiment = build_formation_experiment(**asdict(candidate))

    formation_solution = run_model(
        formation_experiment,
        parameter_values,
        pre_step_last_state,
    )
    formation_time_h = (formation_solution.t[-1] - formation_solution.t[0]) / 3600

    aging_solution = run_model(
        aging_experiment,
        parameter_values,
        formation_solution.last_state,
    )

    result = asdict(candidate)
    result["formation_time_h"] = formation_time_h
    result.update(evaluate_ageing_solution(aging_solution))
    return result


def write_results(path, results):
    if not results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def print_top_results(results, top_n=TOP_N):
    print(f"\nTop {top_n} recommended formation protocols")
    print(
        f"{'Rank':>4} {'Score':>8} {'x1 C':>8} {'x2 C':>8} "
        f"{'Rest min':>9} {'Cycles':>7} {'Form h':>9} "
        f"{'Final cap':>10} {'Cap fade':>10} {'Plate d':>10} "
        f"{'SEI d':>10} {'R d':>10}"
    )
    print("-" * 125)
    for rank, result in enumerate(results[:top_n], start=1):
        print(
            f"{rank:>4} {result['score']:>8.3f} "
            f"{result['x1_charge_rate']:>8.3g} "
            f"{result['x2_discharge_rate']:>8.3g} "
            f"{result['x3_rest_minutes']:>9} "
            f"{result['x4_num_cycles']:>7} "
            f"{result['formation_time_h']:>9.2f} "
            f"{result['capacity_final']:>10.3f} "
            f"{result['capacity_fade']:>10.3f} "
            f"{result['plating_delta']:>10.3f} "
            f"{result['sei_delta']:>10.3f} "
            f"{result['resistance_delta']:>10.3f}"
        )


def main():
    parameter_values = load_parameter_values()
    candidates = list(build_candidates())
    aging_experiment = build_aging_experiment(AGEING_CYCLES)

    print(f"Running pre-step once for {len(candidates)} MCO candidates")
    pre_step_solution = run_model(
        build_pre_step_experiment(),
        parameter_values,
    )

    results = []
    failures = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"\nCandidate {index}/{len(candidates)}: {candidate}")
        start_time = time.time()
        try:
            result = run_candidate(
                candidate,
                parameter_values,
                pre_step_solution.last_state,
                aging_experiment,
            )
        except Exception as error:
            failures.append({**asdict(candidate), "error": str(error)})
            print(f"Failed: {error}")
            continue

        result["runtime_s"] = time.time() - start_time
        results.append(result)
        print(
            f"Completed in {result['runtime_s']:.2f}s | "
            f"formation={result['formation_time_h']:.2f}h | "
            f"final_capacity={result['capacity_final']:.3f}A.h"
        )

    if not results:
        write_results(RESULTS_DIR / "failed_candidates.csv", failures)
        raise RuntimeError("All MCO candidates failed. See failed_candidates.csv.")

    ranked_results = add_scores(results)
    write_results(RESULTS_DIR / "all_candidates.csv", ranked_results)
    write_results(RESULTS_DIR / "top_4_candidates.csv", ranked_results[:TOP_N])
    write_results(RESULTS_DIR / "failed_candidates.csv", failures)

    print_top_results(ranked_results)
    print(f"\nSaved MCO results in {RESULTS_DIR}")
    if failures:
        print(f"Failed candidates: {len(failures)}")


if __name__ == "__main__":
    main()
