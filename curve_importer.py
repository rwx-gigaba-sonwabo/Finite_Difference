import pandas as pd
import os


class CurveImporter:
    """
    Imports .csv from file address for curves
    (3 columns: Used Date, Point Benchmark, Value)
    Uses columns 1 (dates) and 3 (rates), skipping column 2 (tenor)
    """

    def __init__(self):
        self.dates = []
        self.values = []

    def load_data(self, file_path: str):
        """
        Load data from CSV file

        Parameters:
        file_path (str): file path to .csv file

        Expected CSV format:
        Used Date, Point Benchmark, Value
        2025/06/16,,2.00
        2025/06/17,,2.10
        etc.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        # Use first column (dates) and third column (rates)
        date_strings = df.iloc[:, 0].astype(str)
        self.dates = pd.to_datetime(date_strings).tolist()

        # Convert rates from percentages to decimals
        self.values = (df.iloc[:, 2] / 100).tolist()

        return self.dates, self.values
