import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from xverse.transformer import WOEEncoder  # For Weight of Evidence (WOE) encoding
from sklearn.model_selection import train_test_split


class FeatureEngineering:
    def __init__(self, data_path):
        """
        Initialize the Feature Engineering class by loading the dataset.
        :param data_path: Path to the dataset (CSV file).
        """
        self.data = pd.read_csv(data_path)

    def create_aggregate_features(self):
        """
        Create aggregate features such as sum, mean, count, and standard deviation for a numerical variable.
        """
        print("Creating aggregate features...")
        # Example aggregate features
        self.data['Total_Transaction_Amount'] = self.data.groupby('customerId')['Amount'].transform('sum')
        self.data['Average_Transaction_Amount'] = self.data.groupby('customerId')['Amount'].transform('mean')
        self.data['Transaction_Count'] = self.data.groupby('customerId')['Amount'].transform('count')
        self.data['Transaction_StdDev'] = self.data.groupby('customerId')['Amount'].transform('std')
        print("Aggregate features created.")
