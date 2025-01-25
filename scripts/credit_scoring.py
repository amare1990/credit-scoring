import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from xverse.transformer import WOE

class CreditScoring:
    def __init__(self, data_path):
        """
        Initialize the CreditScoring class by loading the dataset.
        :param data_path: Path to the dataset (CSV file).
        """
        self.data = pd.read_csv(data_path)
