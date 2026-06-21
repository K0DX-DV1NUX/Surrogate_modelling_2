import time

from experiments import (
    build_aging_experiment,
    build_capacity_check_experiment,
    build_formation_experiment,
    build_pre_step_experiment,
)
from model import run_model
from outputs import (
    OUTPUTS_OF_INTEREST,
    build_health_diagnostics_table,
    build_results_table,
    measured_checkup_resistance,
    measured_discharge_capacity,
)
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
        x2_discharge_rate=0.01,
        x3_rest_minutes=0.1,
        x4_num_cycles=1,
    )
    sol2 = run_timed_step("formation", formation, parameter_values, sol1.last_state)
    formation_time_s = sol2.t[-1] - sol2.t[1]
    print(f"Total Formation time: {formation_time_s / 3600:.2f} hours\n")

    capacity_check = build_capacity_check_experiment(
        c_rate=0.05,
        upper_voltage=4.1,
        lower_voltage=2.7,
    )
    formation_check = run_timed_step(
        "post-formation capacity check",
        capacity_check,
        parameter_values,
        sol2.last_state,
    )
    capacity_after_formation = measured_discharge_capacity(formation_check)
    resistance_after_formation = measured_checkup_resistance(formation_check)

    num_cycles = 500
    aging = build_aging_experiment(num_cycles)
    sol3 = run_timed_step("aging", aging, parameter_values, sol2.last_state)

    aging_check = run_timed_step(
        "post-aging capacity check",
        capacity_check,
        parameter_values,
        sol3.last_state,
    )
    capacity_after_aging = measured_discharge_capacity(aging_check)
    resistance_after_aging = measured_checkup_resistance(aging_check)

    print(
        "C/20 diagnostic checks\n"
        f"{'State':<24} {'Capacity [A.h]':>16} {'ECM [Ohm]':>14}\n"
        f"{'-' * 24} {'-' * 16} {'-' * 14}\n"
        f"{'After formation':<24} "
        f"{capacity_after_formation:>16.3f} {resistance_after_formation:>14.6f}\n"
        f"{'After aging':<24} "
        f"{capacity_after_aging:>16.3f} {resistance_after_aging:>14.6f}\n"
        f"{'Change':<24} "
        f"{capacity_after_aging - capacity_after_formation:>16.3f} "
        f"{resistance_after_aging - resistance_after_formation:>14.6f}\n"
    )

    print(build_results_table(sol3, OUTPUTS_OF_INTEREST, num_cycles))
    print()
    print(build_health_diagnostics_table(sol3))

    saved_plots = plot_outputs_of_interest(sol3, OUTPUTS_OF_INTEREST)
    print("\nSaved plots:")
    for path in saved_plots:
        print(path)


if __name__ == "__main__":
    main()
