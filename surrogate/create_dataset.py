import argparse

from inputs import ProjectConfig
from simulations import BatterySimulator, ExperimentFactory, ParameterLoader, SolutionAnalyzer
from surrogate.dataset import SurrogateDatasetBuilder
from surrogate.features import TrajectoryFeaturizer


class DatasetCommand:
    def run(self):
        arguments = self._arguments()
        config = ProjectConfig()
        parameters = ParameterLoader(config.parameters_path).load()
        experiments = ExperimentFactory(config.simulation)
        simulator = BatterySimulator(config.simulation, parameters)
        analyzer = SolutionAnalyzer(config.simulation)
        featurizer = TrajectoryFeaturizer(config.surrogate)
        builder = SurrogateDatasetBuilder(
            config.surrogate,
            config,
            experiments,
            simulator,
            analyzer,
            featurizer,
        )
        builder.create(force=arguments.force, limit_candidates=arguments.limit_candidates)

    def _arguments(self):
        parser = argparse.ArgumentParser(description="Create the surrogate dataset")
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--limit-candidates", type=int)
        return parser.parse_args()


if __name__ == "__main__":
    DatasetCommand().run()
