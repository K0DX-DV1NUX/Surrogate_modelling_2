import pybamm


DEFAULT_FORMATION_X1_C_RATE = 0.01
DEFAULT_FORMATION_X2_C_RATE = 0.03
DEFAULT_FORMATION_X3_REST_MINUTES = 1
DEFAULT_FORMATION_X4_CYCLES = 3


def build_pre_step_experiment():
    pre_step_cycle = (
        "Rest for 2 minutes",
        "Charge at 0.05C until 2.0V",
        "Rest for 1440 minutes",
        "Charge at 0.05C for 100 minutes",
        "Charge at 0.333C for 15 minutes",
        "Rest for 4320 minutes",
    )
    return pybamm.Experiment([pre_step_cycle])


def build_formation_cycle(
    x1_charge_rate=DEFAULT_FORMATION_X1_C_RATE,
    x2_discharge_rate=DEFAULT_FORMATION_X2_C_RATE,
    x3_rest_minutes=DEFAULT_FORMATION_X3_REST_MINUTES,
):
    return (
        f"Charge at {x1_charge_rate}C until 4.3V",
        f"Rest for {x3_rest_minutes} minutes",
        f"Discharge at {x2_discharge_rate}C until 2.7V",
    )


def build_formation_experiment(
    x1_charge_rate=DEFAULT_FORMATION_X1_C_RATE,
    x2_discharge_rate=DEFAULT_FORMATION_X2_C_RATE,
    x3_rest_minutes=DEFAULT_FORMATION_X3_REST_MINUTES,
    x4_num_cycles=DEFAULT_FORMATION_X4_CYCLES,
):
    formation_cycle = build_formation_cycle(
        x1_charge_rate=x1_charge_rate,
        x2_discharge_rate=x2_discharge_rate,
        x3_rest_minutes=x3_rest_minutes,
    )
    return pybamm.Experiment([formation_cycle] * x4_num_cycles)


def build_aging_experiment(num_cycles, termination="80% capacity"):
    aging_cycle = (
        "Charge at 1.8C until 4.1V",
        "Discharge at 1.5C until 2.5V",
    )
    return pybamm.Experiment([aging_cycle] * num_cycles, termination=termination)
