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

    def extract_features(self):
        """
        Extract date/time-related features from a TransactionStartTime column.
        """
        print("Extracting features from transaction timestamps...")
        self.data['transaction_hour'] = pd.to_datetime(self.data['TransactionStartTime']).dt.hour
        self.data['transaction_day'] = pd.to_datetime(self.data['TransactionStartTime']).dt.day
        self.data['transaction_month'] = pd.to_datetime(self.data['TransactionStartTime']).dt.month
        self.data['transaction_year'] = pd.to_datetime(self.data['TransactionStartTime']).dt.year
        print("Features extracted.")


    def encode_categorical_variables(self, method="one_hot"):
        """
        Encode categorical variables using One-Hot Encoding or Label Encoding.
        :param method: Encoding method ("one_hot" or "label").
        """
        print(f"Encoding categorical variables using {method}...")
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns

        if method == "one_hot":
            self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)
        elif method == "label":
            le = LabelEncoder()
            for col in categorical_columns:
                self.data[col] = le.fit_transform(self.data[col])
        else:
            print("Invalid encoding method. Choose 'one_hot' or 'label'.")
        print("Categorical variables encoded.")

    def handle_missing_values(self, strategy="mean"):
        """
        Handle missing values in the dataset using imputation or removal.
        :param strategy: Imputation strategy ("mean", "median", "mode") or "remove".
        """
        print(f"Handling missing values using {strategy} strategy...")
        if strategy in ["mean", "median", "most_frequent"]:
            imputer = SimpleImputer(strategy=strategy)
            self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        elif strategy == "remove":
            self.data = self.data.dropna()
        else:
            print("Invalid strategy. Choose 'mean', 'median', 'most_frequent', or 'remove'.")
        print("Missing values handled.")
