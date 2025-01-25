"""
Feature Engineering:
1. Aggregating features
2. Extracting date like features from TransactionStartTime column.
3. handling missing data, outiliers.
4. Encoding categ data
5. Savng data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats


class FeatureEngineering:
    """
    A class that encapsulates the following functionalities.
    1. Aggregating features 2. Extracting date like features from TransactionStartTime column.
    3. handling missing data, outiliers. 4. Encoding categ data 5. Savng data
    """

    def __init__(self, data=None, data_path=None):
        """
        Initialize the Feature Engineering class by loading the dataset.
        :param data: DataFrame (optional) - If provided, it will be used directly.
        :param data_path: Path to the dataset (CSV file).
        """
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError("Either data or data_path must be provided.")

    def create_aggregate_features(self):
        """
        Create aggregate features such as sum, mean, count, and standard deviation for
        the transaction of each customer.
        """
        print("Creating aggregate features...")
        # Example aggregate features
        self.data['Total_Transaction_Amount'] = self.data.groupby('CustomerId')[
            'Amount'].transform('sum')
        self.data['Average_Transaction_Amount'] = self.data.groupby('CustomerId')[
            'Amount'].transform('mean')
        self.data['Transaction_Count'] = self.data.groupby(
            'CustomerId')['Amount'].transform('count')
        self.data['Transaction_StdDev'] = self.data.groupby(
            'CustomerId')['Amount'].transform('std')
        print("Aggregate features created.")

    def extract_features(self):
        """
        Extract date/time-related features from a TransactionStartTime column.
        """
        print("Extracting features from transaction timestamps...")
        self.data['transaction_hour'] = pd.to_datetime(
            self.data['TransactionStartTime']).dt.hour
        self.data['transaction_day'] = pd.to_datetime(
            self.data['TransactionStartTime']).dt.day
        self.data['transaction_month'] = pd.to_datetime(
            self.data['TransactionStartTime']).dt.month
        self.data['transaction_year'] = pd.to_datetime(
            self.data['TransactionStartTime']).dt.year
        print("Features extracted.")

    def encode_categorical_variables(self, method="one_hot"):
        """
        Encode categorical variables using One-Hot Encoding or Label Encoding.
        :param method: Encoding method ("one_hot" or "label").
        """
        print(f"Encoding categorical variables using {method}...")
        categorical_columns = self.data.select_dtypes(
            include=['object', 'category']).columns

        if method == "one_hot":
            self.data = pd.get_dummies(
                self.data, columns=categorical_columns, drop_first=True)
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

        # Handle numerical columns
        if strategy in ["mean", "median"]:
            numerical_cols = self.data.select_dtypes(
                include=["number"]).columns
            imputer = SimpleImputer(strategy=strategy)
            self.data[numerical_cols] = imputer.fit_transform(
                self.data[numerical_cols])

        # Handle categorical columns (mode)
        elif strategy == "most_frequent":  # Mode for categorical features
            categorical_cols = self.data.select_dtypes(
                include=["object", "category"]).columns
            imputer = SimpleImputer(strategy="most_frequent")
            self.data[categorical_cols] = imputer.fit_transform(
                self.data[categorical_cols])

        # Handle "remove" strategy (removes rows with missing values)
        elif strategy == "remove":
            self.data = self.data.dropna()

        else:
            print(
                "Invalid strategy. Choose 'mean', 'median', 'most_frequent', or 'remove'.")

        print("Missing values handled.")

    def handle_outliers(self, method="iqr", factor=1.5):
        """
        Handle outliers in numerical columns using the specified method.
        :param method: Method to handle outliers ("iqr" or "z_score").
        :param factor: The multiplier for the IQR to define outlier thresholds (default is 1.5).
        """
        print(f"Handling outliers using the {method.upper()} method...")
        numerical_columns = self.data.select_dtypes(
            include=['float64', 'int64']).columns

        if method == "iqr":
            for col in numerical_columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                # Filter out outliers
                self.data = self.data[(self.data[col] >= lower_bound) & (
                    self.data[col] <= upper_bound)]
        elif method == "z_score":
            self.data = self.data[(np.abs(stats.zscore(
                self.data[numerical_columns])) < factor).all(axis=1)]
        else:
            print("Invalid method. Choose 'iqr' or 'z_score'.")
        print("Outliers handled.")

    def normalize_or_standardize(self, method="standardize"):
        """
        Normalize or standardize numerical features.
        :param method: Scaling method ("normalize" or "standardize").
        """
        print(f"{method.capitalize()}ing numerical features...")
        numerical_columns = self.data.select_dtypes(
            include=['float64', 'int64']).columns

        if method == "normalize":
            scaler = MinMaxScaler()
            self.data[numerical_columns] = scaler.fit_transform(
                self.data[numerical_columns])
        elif method == "standardize":
            scaler = StandardScaler()
            self.data[numerical_columns] = scaler.fit_transform(
                self.data[numerical_columns])
        else:
            print("Invalid method. Choose 'normalize' or 'standardize'.")
        print(f"Numerical features {method}d.")

    def save_processed_data(self, output_path):
        """
        Save the processed dataset to a CSV file.
        :param output_path: Path to save the processed dataset.
        """
        self.data.to_csv(output_path, index=False)
        print(f"Processed dataset saved to {output_path}.")
