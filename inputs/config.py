import tomllib
from pathlib import Path


class TomlConfig:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as file:
            self.data = tomllib.load(file)

    def section(self, name):
        return self.data[name]


class ProjectConfig:
    def __init__(self, project_root=None):
        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        inputs_dir = self.project_root / "inputs"
        self.simulation = TomlConfig(inputs_dir / "simulation.toml")
        self.mco = TomlConfig(inputs_dir / "mco.toml")
        self.surrogate = TomlConfig(inputs_dir / "surrogate.toml")
        self.parameters_path = inputs_dir / "parameters.toml"

    def resolve_path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path
