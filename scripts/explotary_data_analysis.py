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
