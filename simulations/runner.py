import time

import pybamm


class BatterySimulator:
    def __init__(self, simulation_config, parameter_values):
        self.config = simulation_config.data
        self.parameter_values = parameter_values

    def build_model(self):
        return pybamm.lithium_ion.DFN(options=self.config["model"]["options"])

    def build_solver(self):
        solver = self.config["solver"]
        return pybamm.IDAKLUSolver(
            rtol=solver["relative_tolerance"],
            atol=solver["absolute_tolerance"],
        )

    def solve(self, experiment, last_state=None):
        simulation = pybamm.Simulation(
            self.build_model(),
            experiment=experiment,
            parameter_values=self.parameter_values,
            solver=self.build_solver(),
        )
        return simulation.solve(starting_solution=last_state)

    def run_solver(self, label, experiment, last_state=None):
        print(f"------Running {label}------")
        started = time.perf_counter()
        solution = self.solve(experiment, last_state)
        elapsed = time.perf_counter() - started
        print(f"{label.capitalize()} elapsed in {elapsed:.2f} seconds\n")
        return solution
