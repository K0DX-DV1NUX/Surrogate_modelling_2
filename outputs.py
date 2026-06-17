import numpy as np


OUTPUTS_OF_INTEREST = [
    (
        "Capacity [A.h]",
        "Discharge capacity [A.h]",
        "capacity.png",
    ),
    (
        "Loss to Negative Li Plating [A.h]",
        "Loss of capacity to negative lithium plating [A.h]",
        "loss_to_negative_li_plating.png",
    ),
    (
        "Loss to SEI [A.h]",
        "Loss of capacity to negative SEI [A.h]",
        "loss_to_sei.png",
    ),
    (
        "Local ECM Resistance [Ohm]",
        "Local ECM resistance [Ohm]",
        "local_ecm_resistance.png",
    ),
]


def cycle_variable_max(cycle, variable_name):
    try:
        return cycle[variable_name].entries.max()
    except KeyError:
        return np.nan


def build_results_table(solution, output_specs, num_cycles):
    output_rows = [
        (
            label,
            cycle_variable_max(solution.cycles[0], variable_name),
            cycle_variable_max(solution.cycles[-1], variable_name),
        )
        for label, variable_name, _ in output_specs
    ]

    table_lines = [
        f"Ageing Results after {num_cycles} cycles:",
        f"{'Metric':<36} {'Start':>12} {'Final':>12}",
        f"{'-' * 36} {'-' * 12} {'-' * 12}",
    ]
    table_lines.extend(
        f"{metric:<36} {start:>12.3f} {final:>12.3f}"
        for metric, start, final in output_rows
    )
    return "\n".join(table_lines)
