import numpy as np
import json

class Standardizer:
    """
    This class computes the mean and standard deviation of the features across all training 
    dataframes and provides methods to standardize the dataframes and inverse transform the 
    predictions. The standardization is done using the formula:
    
    standardized_value = (value - mean) / (std + 1e-8)
    
    The inverse transformation for the targets (SEI Rate and Cell Temperature) is done using 
    the formula:
    original_value = standardized_value * std + mean
    """

    def __init__(self):
        self.stats = {}

    def fit(self, dataframes):

        current = np.concatenate([df["Current [A]"].values for df in dataframes])
        temp    = np.concatenate([df["Cell temperature [K]"].values for df in dataframes])
        voltage = np.concatenate([df["Terminal voltage [V]"].values for df in dataframes])
        sei_r   = np.concatenate([df["SEI Rate"].values for df in dataframes])
        q_cum   = np.concatenate([df["Q_cum"].values for df in dataframes])

        self.stats = {
            "current_mean": current.mean(),
            "current_std": current.std(),

            "temp_mean": temp.mean(),
            "temp_std": temp.std(),

            "voltage_mean": voltage.mean(),
            "voltage_std": voltage.std(),

            "sei_mean": sei_r.mean(),
            "sei_std": sei_r.std(),

            "q_cum_mean": q_cum.mean(),
            "q_cum_std": q_cum.std(),

        }

    def transform(self, df):

        df = df.copy()

        df["Current [A]"] = (
            (df["Current [A]"] - self.stats["current_mean"]) /
            (self.stats["current_std"] + 1e-8)
        )

        df["Cell temperature [K]"] = (
            (df["Cell temperature [K]"] - self.stats["temp_mean"]) /
            (self.stats["temp_std"] + 1e-8)
        )

        df["Terminal voltage [V]"] = (
            (df["Terminal voltage [V]"] - self.stats["voltage_mean"]) /
            (self.stats["voltage_std"] + 1e-8)
        )

        df["SEI Rate"] = (
            (df["SEI Rate"] - self.stats["sei_mean"]) /
            (self.stats["sei_std"] + 1e-8)
        )

        df["Q_cum"] = (
            (df["Q_cum"] - self.stats["q_cum_mean"]) /
            (self.stats["q_cum_std"] + 1e-8)
        )

        return df
    
    def inverse_transform_targets(self, y):
        """
        y_pred shape: (batch, 2)
        Order:
        [sei_rate, temperature]
        """

        y = y.copy()

        y[:, 0] = (
            y[:, 0] * self.stats["sei_std"]
            + self.stats["sei_mean"]
        )

        y[:, 1] = (
            y[:, 1] * self.stats["temp_std"]
            + self.stats["temp_mean"]
        )

        return y
    
    # -------------------------
    # SAVE
    # -------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=4)

    # -------------------------
    # LOAD
    # -------------------------
    def load(self, path):
        with open(path, "r") as f:
            self.stats = json.load(f)