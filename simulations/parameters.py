import importlib
import tomllib

import pybamm


class ParameterLoader:
    def __init__(self, parameter_path):
        self.parameter_path = parameter_path

    def load(self):
        with self.parameter_path.open("rb") as file:
            config = tomllib.load(file)

        parameters = {}
        for entry in config.get("parameter", []):
            parameters[entry["name"]] = self._convert(entry)
        return pybamm.ParameterValues(parameters)

    def _convert(self, entry):
        value_type = entry["type"].strip().lower()
        raw_value = entry["value"]
        if value_type == "function":
            module_name, function_name = raw_value.split(":", maxsplit=1)
            return getattr(importlib.import_module(module_name), function_name)
        if value_type == "list":
            return raw_value
        if value_type == "int":
            return int(raw_value)
        if value_type == "float":
            return float(raw_value)
        if value_type == "str":
            return raw_value
        raise ValueError(f"Unsupported parameter type '{value_type}'")
