import importlib
import tomllib
from pathlib import Path

import pybamm


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
