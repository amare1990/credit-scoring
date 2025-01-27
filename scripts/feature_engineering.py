"""
Feature Engineering:
1. Aggregating features
2. Extracting date like features from TransactionStartTime column.
3. handling missing data, outiliers.
4. Encoding categ data
5. Savng data
"""

# Feature Engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats

import pytz

from datetime import datetime


class FeatureEngineering:
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
        Create aggregate features such as sum, mean, count, and standard deviation for a numerical variable.
        """
        print("Creating aggregate features...")
        # Example aggregate features
        self.data['Total_Transaction_Amount'] = self.data.groupby('CustomerId')['Amount'].transform('sum')
        self.data['Average_Transaction_Amount'] = self.data.groupby('CustomerId')['Amount'].transform('mean')
        self.data['Transaction_Count'] = self.data.groupby('CustomerId')['Amount'].transform('count')
        self.data['Transaction_StdDev'] = self.data.groupby('CustomerId')['Amount'].transform('std')
        print("Aggregate features created.")

    def calculate_recency(self):
        """
        Calculate recency as the time since the most recent transaction for each customer.
        """
        print("Calculating Recency...")

        # Ensure 'TransactionStartTime' is in datetime format
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])

        # Get the most recent transaction time for each customer
        most_recent_transactions = self.data.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
        most_recent_transactions.rename(columns={'TransactionStartTime': 'MostRecentTransaction'}, inplace=True)

        # Merge most recent transaction times back to the main dataset
        self.data = self.data.merge(most_recent_transactions, on='CustomerId', how='left')

        # Convert 'datetime.now()' to the same timezone as 'TransactionStartTime' if it has timezone info
        now = datetime.now(pytz.UTC) if self.data['MostRecentTransaction'].dt.tz is not None else datetime.now()

        # Calculate recency in days
        self.data['Recency'] = (now - self.data['MostRecentTransaction']).dt.days

        print("Recency calculated.")



    def extract_features(self):
        """
        Extract date/time-related features from a TransactionStartTime column.
        """
        print("Extracting features from transaction timestamps...")
        self.data['transaction_hour'] = pd.to_datetime(self.data['TransactionStartTime']).dt.hour
        self.data['transaction_day'] = pd.to_datetime(self.data['TransactionStartTime']).dt.day # day of month
        self.data['transaction_month'] = pd.to_datetime(self.data['TransactionStartTime']).dt.month
        self.data['transaction_year'] = pd.to_datetime(self.data['TransactionStartTime']).dt.year

        self.data['transaction_day_of_week'] = pd.to_datetime(self.data['TransactionStartTime']).dt.dayofweek
        # self.data['is_weekend'] = self.data['transaction_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        print("Features extracted.")

    def handle_missing_values(self, strategy="mean", threshold=0.3):
        """
        Handle missing values in the dataset. Features with missing values above the threshold are dropped.
        Otherwise, missing values are handled using imputation (mean, median, or most_frequent).

        :param strategy: Imputation strategy for features with missing values below the threshold
                        ("mean", "median", "most_frequent").
        :param threshold: Proportion of missing values (between 0 and 1) to determine if a feature should be dropped.
        """
        print(f"Handling missing values using {strategy} strategy with a threshold of {threshold * 100}%...")
        initial_row_count = len(self.data)

        # Calculate the percentage of missing values for each feature
        missing_percentage = self.data.isnull().mean()

        # Identify features to drop and features to impute
        features_to_drop = missing_percentage[missing_percentage > threshold].index
        features_to_impute = missing_percentage[missing_percentage <= threshold].index

        # Drop features exceeding the threshold
        self.data = self.data.drop(columns=features_to_drop)
        print(f"Dropped features: {list(features_to_drop)}")

        # Separate features into numerical and categorical for imputation
        numerical_cols = self.data.select_dtypes(include=["number"]).columns
        categorical_cols = self.data.select_dtypes(include=["object", "category"]).columns
        print(f"Numerical columns missing values to handle on\n: {list(numerical_cols)}")
        print(f"Categorical columns missing values to handle on\n: {list(categorical_cols)}")

        # Handle imputation
        if strategy in ["mean", "median"]:
            if not numerical_cols.empty:
                imputer = SimpleImputer(strategy=strategy)
                self.data[numerical_cols] = imputer.fit_transform(self.data[numerical_cols])
                print("Handled missing values for numerical features.")
            else:
                print("No numerical columns to impute.")

        elif strategy == "most_frequent":
            if not categorical_cols.empty:
                imputer = SimpleImputer(strategy="most_frequent")
                self.data[categorical_cols] = imputer.fit_transform(self.data[categorical_cols])
                print("Handled missing values for categorical features.")
            else:
                print("No categorical columns to impute.")

        else:
            print("Invalid strategy. Choose 'mean', 'median', or 'most_frequent'.")

        print("Missing values handled (for categorical and numerical features).")

        print(f"Missing values handled. Final row count: {len(self.data)}")
        print(f"Total rows dropped after missing value handling: {initial_row_count - len(self.data)}\n")



    def handle_outliers(self, method="iqr", factor=1.5):
        """
        Handle outliers in numerical columns using the specified method and log columns with the most outliers.

        :param method: Method to handle outliers ("iqr" or "z_score").
        :param factor: The multiplier for the IQR to define outlier thresholds (default is 1.5).
        """
        print(f"Handling outliers using the {method.upper()} method...")
        initial_row_count = len(self.data)
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        outlier_counts = {}  # Dictionary to track outliers per column

        if method == "iqr":
            for col in numerical_columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                # Count outliers in the column
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                outlier_counts[col] = len(outliers)

                # Filter out outliers
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]

        elif method == "z_score":
            z_scores = stats.zscore(self.data[numerical_columns])
            outliers = (np.abs(z_scores) > factor)
            outlier_counts = outliers.sum(axis=0).to_dict()  # Sum outliers for each column

            # Filter out rows with outliers in any column
            self.data = self.data[~outliers.any(axis=1)]
        else:
            print("Invalid method. Choose 'iqr' or 'z_score'.")
            return

        # Log outlier information
        print("Outliers handled.")
        sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
        print("\nColumns with the highest number of outliers:")
        for col, count in sorted_outliers:
            print(f"{col}: {count} outliers")


        rows_dropped = initial_row_count - len(self.data)
        print(f"Outliers handled. Rows dropped due to outliers: {rows_dropped}")
        print(f"Final row count: {len(self.data)}\n")



    def encode_categorical_variables(self, method="one_hot"):
        """
        Encode categorical variables using One-Hot Encoding or Label Encoding.
        :param method: Encoding method ("one_hot" or "label").
        """
        print(f"Encoding categorical variables using {method}...")
        # Step 2: Drop unnecessary or high-cardinality features that are unique IDs
        # self.data.drop(['CustomerId', 'SubscriptionId', 'AccountId',
        #         'BatchId', 'TransactionId', 'CurrencyCode',
        #         'TransactionStartTime'], axis=1, inplace=True)

        """
        Drop TransactionStartTime column and CurrencyCode. Currency code is unique of one
        and thus doesnot affect the model performance.
        Since pther date like features have been extracted, TransactionStartTime is unnecessary.
        """
        self.data.drop(['TransactionStartTime', 'CurrencyCode'], axis=1, inplace=True)


        # Step 4: Encode high-cardinality features using frequency encoding
        # for col in ['CustomerId', 'SubscriptionId', 'AccountId']:
        #     self.data[f'{col}_freq'] = self.data[col].map(self.data[col].value_counts())

        # categorical_columns = self.data.select_dtypes(
        #     include=['object', 'category']).columns

         # Step 3: Encode low-cardinality categorical features
        categorical_columns = ['ChannelId', 'ProductCategory', 'ProductId', 'ProviderId']

        if method == "one_hot":
            self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)
            print(f"\n\n=========Categorical variables {categorical_columns} encoded using One-Hot Encoding.======\n\n")
        elif method == "label":
            le = LabelEncoder()
            for col in categorical_columns:
                self.data[col] = le.fit_transform(self.data[col])
        else:
            print("Invalid encoding method. Choose 'one_hot' or 'label'.")
        print("Categorical variables encoded.")

    def normalize_or_standardize(self, method="standardize"):
        """
        Normalize or standardize numerical features.
        :param method: Scaling method ("normalize" or "standardize").
        """
        print(f"{method.capitalize()}ing numerical features...")
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

        if method == "normalize":
            scaler = MinMaxScaler()
            self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
        elif method == "standardize":
            scaler = StandardScaler()
            self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
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
