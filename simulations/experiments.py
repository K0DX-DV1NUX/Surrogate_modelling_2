from dataclasses import dataclass

import pybamm


@dataclass(frozen=True)
class FormationCandidate:
    x1_charge_rate: float
    x2_discharge_rate: float
    x3_rest_minutes: float
    x4_num_cycles: int


class ExperimentFactory:
    def __init__(self, simulation_config):
        self.config = simulation_config.data

    def pre_step(self):
        return pybamm.Experiment([tuple(self.config["experiments"]["pre_step"])])

    def default_formation_candidate(self):
        formation = self.config["formation"]
        return FormationCandidate(
            x1_charge_rate=formation["default_charge_rate"],
            x2_discharge_rate=formation["default_discharge_rate"],
            x3_rest_minutes=formation["default_rest_minutes"],
            x4_num_cycles=formation["default_cycles"],
        )

    def formation(self, candidate=None):
        formation = self.config["formation"]
        candidate = candidate or self.default_formation_candidate()
        cycle = (
            f"Charge at {candidate.x1_charge_rate}C until "
            f"{formation['charge_voltage']}V",
            f"Rest for {candidate.x3_rest_minutes} minutes",
            f"Discharge at {candidate.x2_discharge_rate}C until "
            f"{formation['discharge_voltage']}V",
        )
        return pybamm.Experiment([cycle] * candidate.x4_num_cycles)

    def aging(self, cycles):
        cycle = tuple(self.config["experiments"]["aging"])
        return pybamm.Experiment([cycle] * cycles)

    def capacity_check(self):
        check = self.config["capacity_check"]
        cycle = (
            f"Charge at {check['c_rate']}C until {check['upper_voltage']}V",
            f"Discharge at {check['c_rate']}C until {check['lower_voltage']}V",
        )
        return pybamm.Experiment([cycle])
