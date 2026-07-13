from inputs import ProjectConfig
from surrogate.features import TrajectoryFeaturizer
from surrogate.training import SurrogateTrainer

import logging

logging.basicConfig(
    filename="formation_system.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a",
)

logger = logging.getLogger(__name__)

class TrainingCommand:
    def run(self):
        config = ProjectConfig()
        trainer = SurrogateTrainer(
            config.surrogate,
            config,
            TrajectoryFeaturizer(config.surrogate),
        )
        trainer.train()


if __name__ == "__main__":
    TrainingCommand().run()
