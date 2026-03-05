import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class BuildDataframes:
    """
    
    This class is responsible for loading CSV files from a specified directory,
    validating their structure, and transforming the data into a consistent format
    suitable for machine learning models. It performs the following steps:
    1. Loads all CSV files from the given directory.
    2. Validates that each file contains the required columns and that the time column is strictly increasing.
    3. Converts the "Negative SEI thickness [nm]" column into a rate of change (SEI Rate).
    4. Adds a cumulative feature "Q_cum" which represents the cumulative absolute current in Ah.
    5. Stores the processed dataframes in a list for later retrieval.
    """

    REQUIRED_COLUMNS = [
        "Time [s]",
        "Current [A]",
        "Terminal voltage [V]",
        "Cell temperature [K]",
        "Negative SEI thickness [nm]",
    ]

    NEW_COLUMNS = [
        "Current [A]",
        "Terminal voltage [V]",
        "Cell temperature [K]",
        "SEI Rate",
        "Q_cum",
    ]

    def __init__(self, data_folder):

        self.data_folder = data_folder
        self.dataframes = []
        self._load_csv_files()


    def _load_csv_files(self):
        """
        Load all CSV files from the specified directory, validate their structure, and transform the data.
        """

        if not os.path.isdir(self.data_folder):
            raise NotADirectoryError(
                f"{self.data_folder} is not a valid directory."
            )

        csv_files = [
            f for f in os.listdir(self.data_folder)
            if f.lower().endswith(".csv")
        ]

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_folder}"
            )

        for file_name in csv_files:

            full_path = os.path.join(self.data_folder, file_name)

            df = pd.read_csv(full_path)

            self._validate_columns(df, file_name)
            self._validate_time(df, file_name)

            df = self._rate_conversion(df)
            df = self._add_cumulative_features(df)

            df = df[self.NEW_COLUMNS]

            self.dataframes.append(df)


    def _validate_columns(self, df, file_name):
        """"
        Validate that the dataframe contains all required columns.
        """

        df_columns = list(df.columns)

        missing_cols = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df_columns
        ]

        if missing_cols:
            raise ValueError(
                f"File '{file_name}' is missing required columns: {missing_cols}"
            )


    def _validate_time(self, df, file_name):
        """
        Validate that the "Time [s]" column is strictly increasing.
        """

        time = df["Time [s]"].values

        if not np.all(np.diff(time) > 0):
            raise ValueError(
                f"Time column is not strictly increasing in file '{file_name}'."
            )


    def _rate_conversion(self, df):
        """
        Convert "Negative SEI thickness [nm]" to a rate of change (SEI Rate).
        """

        df = df.copy()

        time = df["Time [s]"].values
        sei = df["Negative SEI thickness [nm]"].values

        dt = np.diff(time)
        dsei = np.diff(sei)

        rate_sei = dsei / (dt + 1e-12)

        rate_sei = np.insert(rate_sei, 0, 0.0)

        # Remove old columns
        df = df.drop(
            columns=[
                "Negative SEI thickness [nm]",
            ]
        )

        df["SEI Rate"] = rate_sei

        return df

    def _add_cumulative_features(self, df):
        """
        Add a cumulative feature "Q_cum" which represents the cumulative absolute current in Ah.
        """

        df = df.copy()

        current = df["Current [A]"].values
        #voltage = df["Terminal voltage [V]"].values
        time = df["Time [s]"].values

        dt_hours = np.diff(time, prepend=time[0]) / 3600.0  # time step in hours

        # Cumulative current in Ah (absolute value)
        Q_cum = np.cumsum(np.abs(current) * dt_hours)
        
        df["Q_cum"] = np.sqrt(Q_cum + 1e-12)  # Add small constant to avoid sqrt of zero
        
        return df

    def get_dataframes(self):
        return self.dataframes


if __name__ == "__main__":
    builder = BuildDataframes("Experiments/test")
    dfs = builder.get_dataframes()
    sei_rates = [df["SEI Rate"].values for df in dfs]
    plt.figure(figsize=(10, 6))
    for i, sei_rate in enumerate(sei_rates):
        plt.plot(sei_rate, label=f"Experiment {i+1}")
    plt.title("SEI")
    plt.xlabel("Time Steps")
    plt.ylabel("SEI")
    plt.legend()
    plt.grid()
    plt.show()