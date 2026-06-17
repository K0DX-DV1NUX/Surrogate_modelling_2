from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from outputs import cycle_variable_max


def plot_outputs_of_interest(solution, output_specs, plots_dir="plots", start_cycle=2):
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    cycle_numbers = np.arange(len(solution.cycles))
    saved_paths = []

    for label, variable_name, filename in output_specs:
        values = [
            cycle_variable_max(cycle, variable_name)
            for cycle in solution.cycles
        ]

        plt.figure(figsize=(8, 5))
        plt.plot(
            cycle_numbers[start_cycle:],
            values[start_cycle:],
            marker="o",
            linewidth=1.5,
            markersize=3,
        )
        plt.xlabel("Cycle")
        plt.ylabel(label)
        plt.title(label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = plots_path / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        saved_paths.append(save_path)

    return saved_paths
