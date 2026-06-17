import pybamm


MODEL_OPTIONS = {
    "open-circuit potential": "single",
    "intercalation kinetics": "symmetric Butler-Volmer",
    "thermal": "lumped",
    "surface form": "algebraic",
    "surface temperature": "lumped",
    "SEI": "ec reaction limited",
    "SEI film resistance": "distributed",
    "SEI porosity change": "false",
    "lithium plating": "reversible",
    "lithium plating porosity change": "false",
}


def build_model():
    return pybamm.lithium_ion.DFN(options=MODEL_OPTIONS)


def build_solver():
    return pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)


def run_model(experiment, parameter_values, last_state=None):
    sim = pybamm.Simulation(
        build_model(),
        experiment=experiment,
        parameter_values=parameter_values,
        solver=build_solver(),
    )
    return sim.solve(starting_solution=last_state if last_state is not None else None)
