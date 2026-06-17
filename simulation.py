import time

from experiments import (
    build_aging_experiment,
    build_formation_experiment,
    build_pre_step_experiment,
)
from model import run_model
from outputs import OUTPUTS_OF_INTEREST, build_results_table
from parameters import load_parameter_values
from plotting import plot_outputs_of_interest


def run_timed_step(label, experiment, parameter_values, last_state=None):
    print(f"------Running {label}------")
    start_time = time.time()
    solution = run_model(experiment, parameter_values, last_state)
    end_time = time.time()
    print(f"{label.capitalize()} experiment elapsed in {end_time - start_time:.2f} seconds\n")
    return solution


def main():
    parameter_values = load_parameter_values()

    print("\n")
    pre_step = build_pre_step_experiment()
    sol1 = run_timed_step("pre-step", pre_step, parameter_values)

    formation = build_formation_experiment(
        x1_charge_rate=0.01,
        x2_discharge_rate=0.03,
        x3_rest_minutes=1,
        x4_num_cycles=3,
    )
    sol2 = run_timed_step("formation", formation, parameter_values, sol1.last_state)
    formation_time_s = sol2.t[-1] - sol2.t[0]
    print(f"Total Formation time: {formation_time_s / 3600:.2f} hours\n")

    num_cycles = 100
    aging = build_aging_experiment(num_cycles)
    sol3 = run_timed_step("aging", aging, parameter_values, sol2.last_state)

    print(build_results_table(sol3, OUTPUTS_OF_INTEREST, num_cycles))

    saved_plots = plot_outputs_of_interest(sol3, OUTPUTS_OF_INTEREST)
    print("\nSaved plots:")
    for path in saved_plots:
        print(path)


if __name__ == "__main__":
    main()
