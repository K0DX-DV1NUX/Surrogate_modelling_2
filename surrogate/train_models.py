from inputs import ProjectConfig
from surrogate.features import TrajectoryFeaturizer
from surrogate.training import SurrogateTrainer


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
