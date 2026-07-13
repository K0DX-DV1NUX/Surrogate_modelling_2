import math

import numpy as np

import logging

logger = logging.getLogger(__name__)

class TrajectoryFeaturizer:
    def __init__(self, surrogate_config):
        self.config = surrogate_config.data
        self.feature_count = self.config["data"]["trajectory_feature_count"]
        self.signals = self.config["signals"]

    def extract(self, formation_solution):
        time_s = self._solution_variable(formation_solution, self.signals["time"])
        voltage = self._solution_variable(formation_solution, self.signals["voltage"])
        current = self._solution_variable(formation_solution, self.signals["current"])
        time_s, voltage, current = self._unique_time(time_s, voltage, current)
        time_h = (time_s - time_s[0]) / 3600
        q_signed = self._cumulative_trapezoid(current, time_s) / 3600

        features = {}
        for name, values in {
            "voltage": voltage,
            "current": current,
            "q_signed": q_signed,
        }.items():
            uniform = self._uniform_time(time_h, values, self.feature_count)
            for index, value in enumerate(uniform):
                features[f"{name}_{index:03d}"] = float(value)
        return features

    def feature_columns(self):
        columns = []
        for name in ("voltage", "current", "q_signed"):
            columns.extend(
                f"{name}_{index:03d}"
                for index in range(self.feature_count)
            )
        columns.append("ageing_cycle")
        return columns

    def _solution_variable(self, solution, name):
        return np.asarray(solution[name].entries, dtype=float).reshape(-1)

    def _unique_time(self, time_s, *series):
        unique_time, indices = np.unique(time_s, return_index=True)
        return [unique_time] + [values[indices] for values in series]

    def _cumulative_trapezoid(self, values, time_s):
        if len(values) < 2:
            return np.zeros_like(values)
        increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time_s)
        return np.concatenate([[0.0], np.cumsum(increments)])

    def _uniform_time(self, time_h, values, count):
        if len(time_h) == 1 or math.isclose(time_h[-1], time_h[0]):
            return np.full(count, values[-1])
        target = np.linspace(time_h[0], time_h[-1], count)
        return np.interp(target, time_h, values)
