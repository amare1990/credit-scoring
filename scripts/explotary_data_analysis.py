import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class ExploratoryDataAnalysis:
    def __init__(self, data_path):
        """
        Initialize the EDA class by loading the dataset.
        :param data_path: Path to the dataset (CSV file).
        """
        self.data = pd.read_csv(data_path)

    def overview_of_data(self):
        """
        Provide an overview of the dataset, including rows, columns, and data types.
        """
        print("Overview of the Data:")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nFirst 5 Rows:")
        print(self.data.head())
        print("\n")

    def summary_statistics(self):
        """
        Display summary statistics for the numerical columns.
        """
        print("Summary Statistics:")
        print(self.data.describe())
        print("\n")

    def summary_statistics(self):
        """
        Display summary statistics for the numerical columns.
        """
        print("Summary Statistics:")
        print(self.data.describe())
        print("\n")
