import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from scorecardpy import woebin
import scorecardpy as sc
from monotonic_binning.monotonic_woe_binning import Binning
from datetime import datetime
from sklearn.model_selection import train_test_split  # Import train_test_split



class CreditScoring:
    def __init__(self, data_path):
        """
        Initialize the CreditScoring class by loading the dataset.
        :param data_path: Path to the dataset (CSV file).
        """
        self.data = pd.read_csv(data_path)

    def calculate_rfms(self, recency_col, frequency_col, monetary_col):
        """
        Calculate the RFMS score for each user.
        RFMS Score = Weighted combination of Recency, Frequency, and Monetary values.
        :param recency_col: Column representing recency (time since last transaction).
        :param frequency_col: Column representing frequency (number of transactions).
        :param monetary_col: Column representing monetary value (total transaction amount).
        """
        print("Calculating RFMS Score...")
        # Normalize recency, frequency, and monetary
        self.data['Recency_Score'] = 1 / (1 + self.data[recency_col])
        self.data['Frequency_Score'] = self.data[frequency_col] / self.data[frequency_col].max()
        self.data['Monetary_Score'] = self.data[monetary_col] / self.data[monetary_col].max()

        # Calculate RFMS score as a weighted sum
        self.data['RFMS_Score'] = (
            0.4 * self.data['Recency_Score'] +
            0.3 * self.data['Frequency_Score'] +
            0.3 * self.data['Monetary_Score']
        )
        print(self.data[['Recency_Score', 'Frequency_Score', 'Monetary_Score', 'RFMS_Score']].head())


    def visualize_rfms_distribution(self, rfms_score_col):
        """
        Visualize the RFMS score distribution and the threshold boundary.
        :param rfms_score_col: Column containing the RFMS score.
        """
        print("Visualizing RFMS Score Distribution...")
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data[rfms_score_col], kde=True, bins=30, color='skyblue')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.title('RFMS Score Distribution')
        plt.xlabel('RFMS Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()



    def classify_users(self, rfms_score_col, threshold=0.5):
        """
        Classify users as "good" or "bad" based on their RFMS score.
        :param rfms_score_col: Column containing the RFMS score.
        :param threshold: Threshold to classify users into "good" (>= threshold) and "bad" (< threshold).
        """
        print("Classifying Users into Good and Bad...")
        self.data['Creditworthiness'] = np.where(self.data[rfms_score_col] >= threshold, 'Good', 'Bad')
        print(self.data[['RFMS_Score', 'Creditworthiness']].head())


    def apply_woe_binning_monotonic(self, target_col, exclude_cols=None):
        """
        Applies monotonic WoE binning to numeric variables and processes categorical variables for WoE transformation.

        Args:
            target_col (str): The name of the target column.
            exclude_cols (list, optional): A list of columns to exclude from binning. Defaults to None.

        Returns:
            dict: A dictionary with binning breaks for both numeric and categorical variables.
        """
        print("Applying Monotonic WoE Binning...")
        df = self.data
        # Split into training and testing datasets
        train, test = sc.split_df(df, target_col, ratio=0.7, seed=999).values()
        exclude_cols = exclude_cols or []

        # Convert 'Creditworthiness' to numerical *before* binning
        df[target_col] = df[target_col].map({'Good': 1, 'Bad': 0})
        train[target_col] = train[target_col].map({'Good': 1, 'Bad': 0})  # Convert in train as well
        test[target_col] = test[target_col].map({'Good': 1, 'Bad': 0})    # Convert in test as well

        # Separate numeric and categorical variables
        numeric_vars = train.select_dtypes(include=['float64', 'int64']).columns.difference([target_col] + exclude_cols)

        # WoE binning and calculate WoE
        categorical_vars = train[['ChannelId', 'ProductCategory', 'ProductId', 'ProviderId']]

        # Initialize binning breaks dictionary
        breaks = {}

        # Numeric variables: Monotonic WoE binning
        def woe_numeric(x, y):
            bin_object = Binning(y, n_threshold=50, y_threshold=10, p_threshold=0.35, sign=False)
            for feature in x:
                print(f"Binning numeric feature: {feature}")
                bin_object.fit(train[[y, feature]])
                breaks[feature] = bin_object.bins[1:-1].tolist()

        woe_numeric(numeric_vars, target_col)

        print("Monotonic WoE binning on numerical columns completed.")

        # Initialize cat_breaks to avoid error when there are no categorical variables
        cat_breaks = {}

        # Categorical variables: WoE binning using scorecardpy
        if len(categorical_vars) > 0:
            print("Processing categorical variables...")
            cat_breaks = sc.woebin(train, y=target_col, x=list(categorical_vars), save_breaks_list='cat_breaks')

        # Merge numeric and categorical variable breaks
        print("Merging binning breaks for both numeric and categorical variables...")
        bins = {**breaks, **cat_breaks}  # Merge the breaks of both numeric and categorical variables

        # Compute WoE for all features using merged breaks
        print("Computing WoE for all features...")
        bins_adj = sc.woebin(df, y=target_col, breaks_list=bins, positive='bad|0')

        return bins_adj  # Return the final WoE adjusted bins




    def visualize_woe_binning(self, bins, feature_col):
        """
        Visualize the WoE values of a feature.
        :param bins: The WoE binning results from scorecardpy.
        :param feature_col: Feature column to visualize the WoE values for.
        """
        print(f"\n\nVisualizing WoE Binning Results for {feature_col}...")

        # Convert the bins list to a DataFrame suitable for seaborn
        woe_df = pd.DataFrame({'bin': bins[feature_col], 'woe': range(len(bins[feature_col]))}) # Creating a dummy woe for visualization
                                                                                                # Replace with actual woe calculation if needed

        plt.figure(figsize=(8, 6))
        sns.barplot(x='bin', y='woe', data=woe_df)
        plt.title(f'WoE Values for {feature_col}')
        plt.xlabel('Bins')
        plt.ylabel('Weight of Evidence (WoE)')
        plt.show()
