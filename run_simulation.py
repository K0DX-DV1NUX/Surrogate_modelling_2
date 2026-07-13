from inputs import ProjectConfig
from simulations import (
    BatterySimulator,
    ExperimentFactory,
    ParameterLoader,
    ResultPlotter,
    SolutionAnalyzer,
)

import logging

logging.basicConfig(
    filename="formation_system.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a",
)

logger = logging.getLogger(__name__)


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

        logger.info(f"Formation choice: {candidate}\n")
        
        pre_step = simulator.run_solver(
            label="pre-step", 
            experiment=experiments.pre_step(),
            last_state=None
            )
        
        formation = simulator.run_solver(
            label="formation",
            experiment=experiments.formation(candidate),
            last_state=pre_step.last_state,
        )
        formation_time_h = (formation.t[-1] - formation.t[0]) / 3600
        logger.info(f"Formation simulation time: {formation_time_h:.2f} hours\n")

        capacity_check = experiments.capacity_check()
        formation_check = simulator.run_solver(
            label="post-formation C/20 check",
            experiment=capacity_check,
            last_state=formation.last_state,
        )

        aging_cycles = run_config["aging_cycles"]
        aging = simulator.run_solver(
            label="aging",
            experiment=experiments.aging(aging_cycles),
            last_state=formation.last_state,
        )

        aging_check = simulator.run_solver(
            label="post-aging C/20 check",
            experiment=capacity_check,
            last_state=aging.last_state,
        )

        self._print_checkup_table(analyzer, formation_check, aging_check)
        logger.info(analyzer.results_table(aging, aging_cycles))
        logger.info("")
        logger.info(analyzer.diagnostics_table(aging))

        plots_dir = self.config.resolve_path(run_config["plots_dir"])
        saved_plots = ResultPlotter(analyzer, plots_dir).plot_outputs(aging)
        logger.info("\nSaved plots:")
        for path in saved_plots:
            logger.info(path)

    def _print_checkup_table(self, analyzer, formation_check, aging_check):
        formation_capacity = analyzer.measured_capacity(formation_check)
        aging_capacity = analyzer.measured_capacity(aging_check)
        formation_resistance = analyzer.measured_resistance(formation_check)
        aging_resistance = analyzer.measured_resistance(aging_check)
        logger.info(
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
    try:
        IndividualSimulationRunner().run()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        