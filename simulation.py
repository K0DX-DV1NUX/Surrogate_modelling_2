import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybamm
import importlib
import tomllib
from pathlib import Path
import time


def load_parameter_values(config_path="parameters.toml"):
    parameter_file = Path(config_path)
    try:
        with parameter_file.open("rb") as file:
            config = tomllib.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not read parameter file: {parameter_file}")

    parameters = {}
    for entry in config.get("parameter", []):
        name = entry["name"]
        value_type = entry["type"].strip().lower()
        raw_value = entry["value"]

        if value_type == "function":
            module_name, function_name = raw_value.split(":", maxsplit=1)
            value = getattr(importlib.import_module(module_name), function_name)
        elif value_type == "list":
            value = raw_value
        elif value_type == "int":
            value = int(raw_value)
        elif value_type == "float":
            value = float(raw_value)
        elif value_type == "str":
            value = raw_value
        else:
            raise ValueError(f"Unsupported parameter type '{value_type}' for '{name}'")

        parameters[name] = value

    return pybamm.ParameterValues(parameters)


def cycle_variable_max(cycle, variable_name):
    try:
        return cycle[variable_name].entries.max()
    except KeyError:
        return np.nan


def plotter(solution, output_specs, plots_dir="plots"):
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    cycle_numbers = np.arange(len(solution.cycles))
    saved_paths = []

    for label, variable_name, filename in output_specs:
        values = []
        for cycle in solution.cycles:
            values.append(cycle_variable_max(cycle, variable_name))

        plt.figure(figsize=(8, 5))
        plt.plot(cycle_numbers[2:], values[2:], marker="o", linewidth=1.5, markersize=3)
        plt.xlabel("Cycle")
        plt.ylabel(label)
        plt.title(label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = plots_path / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        saved_paths.append(save_path)

    return saved_paths


def main():

    ## Currently for MCO use Mohtat2020 parameters. 
    # These will be later changed for the CPI batteries.

    parameter_values = load_parameter_values()

    ## Define the Model.
    def run_model(experiment, parameter_values, last_state):
        model = pybamm.lithium_ion.DFN(options={
            "open-circuit potential": "single",
            "intercalation kinetics": "symmetric Butler-Volmer",
            "thermal": "lumped",
            "surface form": "algebraic",
            "surface temperature": "lumped",
            "SEI": "ec reaction limited",
            "SEI film resistance": "distributed",
            "SEI porosity change": "false",
            "lithium plating": "reversible",
            "lithium plating porosity change": "false",
            }
        )

        solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)

        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values, solver=solver)
        sol = sim.solve(starting_solution=last_state if last_state is not None else None)

        return sol

    # -----------------------------
    # PRE-STEP PROCESS (These steps are fixed.)
    # -----------------------------
    pre_step_cycle = (
        "Rest for 2 minutes",
        "Charge at 0.05C until 2.0V",
        "Rest for 1440 minutes",
        "Charge at 0.05C for 100 minutes",
        "Charge at 0.333C for 15 minutes",
        "Rest for 4320 minutes"
    )
    print("\n")
    print(f"------Running pre-step------")
    start_time = time.time()
    pre_step = pybamm.Experiment([pre_step_cycle])
    sol1 = run_model(pre_step, parameter_values, None)
    end_time = time.time()
    print(f"Pre-step experiment elapsed in {end_time - start_time:.2f} seconds\n")

    ## Activate the below to see all keys available in a cycle. 
    ## This is useful to identify the keys for the desired outputs.

    # for key in sol1.all_models[0].variables.keys():
    #     if "porosity" in key.lower():
    #         print(key)


    # -----------------------------
    # FORMATION PROCESS
    # -----------------------------
    formation_cycle = (
        "Charge at 0.01C until 4.3V",
        "Rest for 1 minutes",
        "Discharge at 0.03C until 2.7V",
        )
    print(f"------Running formation------")
    formation = pybamm.Experiment([formation_cycle] * 3)
    start_time = time.time()
    sol2 = run_model(formation, parameter_values, sol1.last_state)
    end_time = time.time()
    print(f"Formation experiment elapsed in {end_time - start_time:.2f} seconds\n")
    formation_time_s = sol2.t[-1] - sol2.t[0]
    print(f"Total Formation time: {formation_time_s / 3600:.2f} hours\n")

    
    # -----------------------------
    # AGEING PROCESS (This step is fixed.)
    # -----------------------------
    aging_cycle = (
        "Charge at 1C until 4.1V",
        "Discharge at 1.1C until 2.7V",
    )
    num_cycles = 100
    aging = pybamm.Experiment([aging_cycle] * num_cycles, termination="80% capacity")
    print(f"------Running aging------")
    start_time = time.time()
    sol3 = run_model(aging, parameter_values, sol2.last_state)
    end_time = time.time()
    print(f"Aging experiment elapsed in {end_time - start_time:.2f} seconds\n")


    outputs_of_interest = [
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

    ## Print the Outputs of interest.
    output_rows = [
        (
            outputs_of_interest[0][0],
            cycle_variable_max(sol3.cycles[0], outputs_of_interest[0][1]),
            cycle_variable_max(sol3.cycles[-1], outputs_of_interest[0][1]),
        ),
        (
            outputs_of_interest[1][0],
            cycle_variable_max(sol3.cycles[0], outputs_of_interest[1][1]),
            cycle_variable_max(sol3.cycles[-1], outputs_of_interest[1][1]),
        ),
        (
            outputs_of_interest[2][0],
            cycle_variable_max(sol3.cycles[0], outputs_of_interest[2][1]),
            cycle_variable_max(sol3.cycles[-1], outputs_of_interest[2][1]),
        ),
        (
            outputs_of_interest[3][0],
            cycle_variable_max(sol3.cycles[0], outputs_of_interest[3][1]),
            cycle_variable_max(sol3.cycles[-1], outputs_of_interest[3][1]),
        ),
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
    print("\n".join(table_lines))

    saved_plots = plotter(sol3, outputs_of_interest)
    print("\nSaved plots:")
    for path in saved_plots:
        print(path)


if __name__ == "__main__":
    main()
