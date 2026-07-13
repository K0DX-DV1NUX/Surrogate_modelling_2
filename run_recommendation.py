from inputs import ProjectConfig
from mco import FormationOptimizer
from simulations import BatterySimulator, ExperimentFactory, ParameterLoader, SolutionAnalyzer
from surrogate import SurrogatePredictor, TrajectoryFeaturizer

import logging

logging.basicConfig(
    filename="recommendations.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a",
)

logger = logging.getLogger(__name__)

class RecommendationRunner:
    def __init__(self):
        self.config = ProjectConfig()

    def run(self):
        parameter_values = ParameterLoader(self.config.parameters_path).load()
        experiments = ExperimentFactory(self.config.simulation)
        simulator = BatterySimulator(self.config.simulation, parameter_values)
        analyzer = SolutionAnalyzer(self.config.simulation)
        predictor = SurrogatePredictor(self.config.surrogate, self.config)
        featurizer = TrajectoryFeaturizer(self.config.surrogate)
        results_dir = self.config.resolve_path(
            self.config.mco.section("recommendation")["results_dir"]
        )

        optimizer = FormationOptimizer(
            self.config.mco,
            experiments,
            simulator,
            analyzer,
            results_dir,
            surrogate_predictor=predictor,
            trajectory_featurizer=featurizer,
        )
        return optimizer.run()


if __name__ == "__main__":
    RecommendationRunner().run()
