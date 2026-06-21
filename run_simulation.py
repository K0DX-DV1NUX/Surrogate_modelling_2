from inputs import ProjectConfig
from simulations import (
    BatterySimulator,
    ExperimentFactory,
    ParameterLoader,
    ResultPlotter,
    SolutionAnalyzer,
)


class IndividualSimulationRunner:
    def __init__(self):
        self.config = ProjectConfig()

    def run(self):
        parameter_values = ParameterLoader(self.config.parameters_path).load()
        experiments = ExperimentFactory(self.config.simulation)
        simulator = BatterySimulator(self.config.simulation, parameter_values)
        analyzer = SolutionAnalyzer(self.config.simulation)
        run_config = self.config.simulation.section("run")
        candidate = experiments.default_formation_candidate()

        print(f"Formation choice: {candidate}\n")
        pre_step = simulator.solve_timed("pre-step", experiments.pre_step())
        formation = simulator.solve_timed(
            "formation",
            experiments.formation(candidate),
            pre_step.last_state,
        )
        formation_time_h = (formation.t[-1] - formation.t[0]) / 3600
        print(f"Formation simulation time: {formation_time_h:.2f} hours\n")

        capacity_check = experiments.capacity_check()
        formation_check = simulator.solve_timed(
            "post-formation C/20 check",
            capacity_check,
            formation.last_state,
        )

        aging_cycles = run_config["aging_cycles"]
        aging = simulator.solve_timed(
            "aging",
            experiments.aging(aging_cycles),
            formation.last_state,
        )
        aging_check = simulator.solve_timed(
            "post-aging C/20 check",
            capacity_check,
            aging.last_state,
        )

        self._print_checkup_table(analyzer, formation_check, aging_check)
        print(analyzer.results_table(aging, aging_cycles))
        print()
        print(analyzer.diagnostics_table(aging))

        plots_dir = self.config.resolve_path(run_config["plots_dir"])
        saved_plots = ResultPlotter(analyzer, plots_dir).plot_outputs(aging)
        print("\nSaved plots:")
        for path in saved_plots:
            print(path)

    def _print_checkup_table(self, analyzer, formation_check, aging_check):
        formation_capacity = analyzer.measured_capacity(formation_check)
        aging_capacity = analyzer.measured_capacity(aging_check)
        formation_resistance = analyzer.measured_resistance(formation_check)
        aging_resistance = analyzer.measured_resistance(aging_check)
        print(
            "C/20 diagnostic checks\n"
            f"{'State':<24} {'Capacity [A.h]':>16} {'ECM [Ohm]':>14}\n"
            f"{'-' * 24} {'-' * 16} {'-' * 14}\n"
            f"{'After formation':<24} "
            f"{formation_capacity:>16.3f} {formation_resistance:>14.6f}\n"
            f"{'After aging':<24} "
            f"{aging_capacity:>16.3f} {aging_resistance:>14.6f}\n"
            f"{'Change':<24} "
            f"{aging_capacity - formation_capacity:>16.3f} "
            f"{aging_resistance - formation_resistance:>14.6f}\n"
        )


if __name__ == "__main__":
    IndividualSimulationRunner().run()
