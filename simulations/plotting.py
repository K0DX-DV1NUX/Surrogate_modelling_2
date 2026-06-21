from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ResultPlotter:
    def __init__(self, analyzer, plots_dir):
        self.analyzer = analyzer
        self.plots_dir = Path(plots_dir)

    def plot_outputs(self, solution):
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for spec in self.analyzer.output_specs:
            values = self.analyzer.metric_series(solution, spec["key"])
            cycle_numbers = np.arange(1, len(values) + 1)
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(cycle_numbers, values, linewidth=1.5)
            axis.set_xlabel("Cycle")
            axis.set_ylabel(spec["label"])
            axis.set_title(spec["label"])
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            path = self.plots_dir / spec["filename"]
            figure.savefig(path, dpi=300)
            plt.close(figure)
            saved.append(path)
        return saved
