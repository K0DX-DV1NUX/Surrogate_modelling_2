import numpy as np


OUTPUTS_OF_INTEREST = [
    (
        "Discharge Capacity [A.h]",
        "Discharge capacity [A.h]",
        "capacity.png",
    ),
    (
        "Loss of capacity to negative lithium plating [A.h]",
        "Loss of capacity to negative lithium plating [A.h]",
        "loss_to_negative_li_plating.png",
    ),
    (
        "Loss of capacity to negative SEI [A.h]",
        "Loss of capacity to negative SEI [A.h]",
        "loss_to_sei.png",
    ),
    (
        "Local ECM Resistance [Ohm]",
        "Local ECM resistance [Ohm]",
        "local_ecm_resistance.png",
    ),
]

HEALTH_DIAGNOSTICS = [
    (
        "Negative active material volume fraction",
        "X-averaged negative electrode active material volume fraction",
    ),
    (
        "Positive active material volume fraction",
        "X-averaged positive electrode active material volume fraction",
    ),
    (
        "Negative electrode porosity",
        "X-averaged negative electrode porosity",
    ),
    (
        "Positive electrode porosity",
        "X-averaged positive electrode porosity",
    ),
    (
        "Negative plated lithium [mol.m-3]",
        "X-averaged negative lithium plating concentration [mol.m-3]",
    ),
    (
        "Cell temperature [K]",
        "Volume-averaged cell temperature [K]",
    ),
]


def cycle_variable_end(cycle, variable_name):
    try:
        return float(cycle[variable_name].entries[-1])
    except KeyError:
        return np.nan


def cycle_discharge_capacity(cycle):
    discharge_step = cycle.steps[-1]
    discharge_capacity = discharge_step["Discharge capacity [A.h]"].entries
    return float(discharge_capacity[-1] - discharge_capacity[0])


def cycle_metric_value(cycle, variable_name):
    if variable_name == "Discharge capacity [A.h]":
        return cycle_discharge_capacity(cycle)
    return cycle_variable_end(cycle, variable_name)


def experiment_cycles(solution):
    cycles = []
    for cycle in solution.cycles:
        time = cycle["Time [s]"].entries
        if len(time) > 1 and time[-1] > time[0]:
            cycles.append(cycle)

    if not cycles:
        raise ValueError("Solution does not contain a nonzero-duration experiment cycle")
    return cycles


def measured_discharge_capacity(capacity_check_solution):
    capacity_check_cycle = experiment_cycles(capacity_check_solution)[-1]
    return cycle_discharge_capacity(capacity_check_cycle)


def measured_checkup_resistance(capacity_check_solution):
    capacity_check_cycle = experiment_cycles(capacity_check_solution)[-1]
    discharge_step = capacity_check_cycle.steps[-1]
    return float(discharge_step["Local ECM resistance [Ohm]"].entries[-1])


def build_results_table(solution, output_specs, num_cycles):
    cycles = experiment_cycles(solution)
    output_rows = [
        (
            label,
            cycle_metric_value(cycles[0], variable_name),
            cycle_metric_value(cycles[-1], variable_name),
        )
        for label, variable_name, _ in output_specs
    ]

    table_lines = [
        f"Ageing Results after {num_cycles} cycles:",
        f"{'Metric':<50} {'Start':>12} {'Final':>12}",
        f"{'-' * 50} {'-' * 12} {'-' * 12}",
    ]
    table_lines.extend(
        f"{metric:<50} {start:>12.3f} {final:>12.3f}"
        for metric, start, final in output_rows
    )
    return "\n".join(table_lines)


def build_health_diagnostics_table(solution):
    cycles = experiment_cycles(solution)
    rows = [
        (
            label,
            cycle_variable_end(cycles[0], variable_name),
            cycle_variable_end(cycles[-1], variable_name),
        )
        for label, variable_name in HEALTH_DIAGNOSTICS
    ]

    table_lines = [
        "Physical-state diagnostics:",
        f"{'Metric':<48} {'Start':>14} {'Final':>14}",
        f"{'-' * 48} {'-' * 14} {'-' * 14}",
    ]
    table_lines.extend(
        f"{metric:<48} {start:>14.6g} {final:>14.6g}"
        for metric, start, final in rows
    )

    warnings = []
    for label, start, final in rows:
        if "porosity" in label.lower() or "volume fraction" in label.lower():
            if not 0 <= final <= 1:
                warnings.append(f"{label} is outside [0, 1]: {final:.6g}")
        if "plated lithium" in label.lower() and final < 0:
            warnings.append(f"{label} is negative: {final:.6g}")

    if warnings:
        table_lines.append("\nPhysical-state warnings:")
        table_lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(table_lines)
