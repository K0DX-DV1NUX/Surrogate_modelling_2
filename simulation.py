import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pybamm

def main():

    ## Currently for MCO use Mohtat2020 parameters. 
    # These will be later changed for the CPI batteries.

    parameter_values = pybamm.ParameterValues("Mohtat2020")
    parameter_values.update(
        {

            # Thermal-lumped and surface temperature-lumped
            #"Cell thermal capacity [J.K-1]": 1200.0,
            "Casing heat capacity [J.K-1]": 300.0,
            "Environment thermal resistance [K.W-1]": 2.0,
            #"Internal thermal resistance [K.W-1]": 0.2,

            # Lithium plating parameters
            "Lithium plating transfer coefficient": 0.5,
            # "Dead Lithium decay constants [s-1]": 1e-4,
            "Exchange-current density for stripping [A.m-2]": 0.1,
            #"Dead lithium decay rate [s-1]": 1e-4
        },
        check_already_exists=False
    )

    ## Define the Model.
    def run_model(experiment, parameter_values, last_state):
        model = pybamm.lithium_ion.DFN(options={
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
        )

        solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)

        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values, solver=solver)
        sol = sim.solve(starting_solution=last_state if last_state is not None else None)

        return sol


    ## These steps are fixed.
    pre_step_cycle = (
        "Rest for 2 minutes",
        "Charge at 0.05C until 2.0V",
        "Rest for 1440 minutes",
        "Charge at 0.05C for 100 minutes",
        "Charge at 0.333C for 15 minutes",
        "Rest for 4320 minutes"
    )

    print("Running pre-step...")
    pre_step = pybamm.Experiment([pre_step_cycle])
    sol1 = run_model(pre_step, parameter_values, None)

    ## Activate the below to see all keys available in a cycle. 
    ## This is useful to identify the keys for the desired outputs.

    # for key in sol1.all_models[0].variables.keys():
    #     if "porosity" in key.lower():
    #         print(key)

    formation_cycle = (
        "Charge at 0.01C until 4.3V", # Charge at x1C until x2V
        "Rest for 1 minutes", # Rest for x3 minutes
        "Discharge at 0.03C until 2.7V",
        ) # 1 min to run

    print("Running formation...")
    formation = pybamm.Experiment([formation_cycle] * 3)
    sol2 = run_model(formation, parameter_values, sol1.last_state)

    aging_cycle = (
        "Charge at 1C until 4.1V",
        "Discharge at 1.1C until 2.7V",
    ) # 2-5 mins to run.

    aging = pybamm.Experiment([aging_cycle] * 1000, termination="80% capacity")

    print("Running aging...")
    sol3 = run_model(aging, parameter_values, sol2.last_state)


    ## Print the Outputs of interest.
    print(f"Capacity\nStart: {sol3.cycles[2]['Discharge capacity [A.h]'].entries.max():.3f}", f"Final: {sol3.cycles[-1]['Discharge capacity [A.h]'].entries.max():.3f}\n")
    print(f"Loss to Negative Li Plating\nStart: {sol3.cycles[2]['Loss of capacity to negative lithium plating [A.h]'].entries.max():.3f}", f"Final: {sol3.cycles[-1]['Loss of capacity to negative lithium plating [A.h]'].entries.max():.3f}\n")
    print(f"Loss to SEI\nStart: \t{sol3.cycles[2][ 'Loss of capacity to negative SEI [A.h]'].entries.max():.3f}", f"Final: {sol3.cycles[-1]['Loss of capacity to negative SEI [A.h]'].entries.max():.3f}\n")
    print(f"Local ECM Resistance\nStart: {sol3.cycles[2]['Local ECM resistance [Ohm]'].entries.max():.3f}", f"Final: {sol3.cycles[-1]['Local ECM resistance [Ohm]'].entries.max():.3f}\n")


if __name__ == "__main__":
    main()