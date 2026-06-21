import numpy as np


class SolutionAnalyzer:
    def __init__(self, simulation_config):
        self.config = simulation_config.data
        self.output_specs = self.config["outputs"]
        self.diagnostic_specs = self.config["diagnostics"]
        self.output_by_key = {spec["key"]: spec for spec in self.output_specs}

    def experiment_cycles(self, solution):
        cycles = []
        for cycle in solution.cycles:
            time = cycle["Time [s]"].entries
            if len(time) > 1 and time[-1] > time[0]:
                cycles.append(cycle)
        if not cycles:
            raise ValueError("Solution contains no nonzero-duration experiment cycle")
        return cycles

    def variable_end(self, cycle, variable_name):
        try:
            return float(cycle[variable_name].entries[-1])
        except KeyError:
            return np.nan

    def discharge_capacity(self, cycle):
        discharge_step = cycle.steps[-1]
        capacity = discharge_step["Discharge capacity [A.h]"].entries
        return float(capacity[-1] - capacity[0])

    def metric_value(self, cycle, output_spec):
        if output_spec["aggregation"] == "discharge_step":
            return self.discharge_capacity(cycle)
        return self.variable_end(cycle, output_spec["variable"])

    def metric_series(self, solution, key):
        spec = self.output_by_key[key]
        return [
            self.metric_value(cycle, spec)
            for cycle in self.experiment_cycles(solution)
        ]

    def first_and_last_metrics(self, solution):
        values = {}
        for spec in self.output_specs:
            series = [value for value in self.metric_series(solution, spec["key"]) if np.isfinite(value)]
            if not series:
                raise ValueError(f"No finite values for {spec['variable']}")
            values[spec["key"]] = (series[0], series[-1])
        return values

    def measured_capacity(self, check_solution):
        return self.discharge_capacity(self.experiment_cycles(check_solution)[-1])

    def measured_resistance(self, check_solution):
        cycle = self.experiment_cycles(check_solution)[-1]
        return self.variable_end(cycle.steps[-1], self.output_by_key["resistance"]["variable"])

    def results_table(self, solution, cycle_count):
        metrics = self.first_and_last_metrics(solution)
        lines = [
            f"Ageing results after {cycle_count} cycles:",
            f"{'Metric':<50} {'Start':>12} {'Final':>12}",
            f"{'-' * 50} {'-' * 12} {'-' * 12}",
        ]
        for spec in self.output_specs:
            start, final = metrics[spec["key"]]
            lines.append(f"{spec['label']:<50} {start:>12.3f} {final:>12.3f}")
        return "\n".join(lines)

    def diagnostics_table(self, solution):
        cycles = self.experiment_cycles(solution)
        rows = []
        warnings = []
        for spec in self.diagnostic_specs:
            start = self.variable_end(cycles[0], spec["variable"])
            final = self.variable_end(cycles[-1], spec["variable"])
            rows.append((spec["label"], start, final))
            low, high = spec["physical_range"]
            if np.isfinite(final) and not low <= final <= high:
                warnings.append(f"{spec['label']} outside [{low}, {high}]: {final:.6g}")

        lines = [
            "Physical-state diagnostics:",
            f"{'Metric':<48} {'Start':>14} {'Final':>14}",
            f"{'-' * 48} {'-' * 14} {'-' * 14}",
        ]
        lines.extend(
            f"{label:<48} {start:>14.6g} {final:>14.6g}"
            for label, start, final in rows
        )
        if warnings:
            lines.append("\nPhysical-state warnings:")
            lines.extend(f"- {warning}" for warning in warnings)
        return "\n".join(lines)
