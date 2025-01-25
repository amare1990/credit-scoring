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

    def classify_users(self, rfms_score_col, threshold=0.5):
        """
        Classify users as "good" or "bad" based on their RFMS score.
        :param rfms_score_col: Column containing the RFMS score.
        :param threshold: Threshold to classify users into "good" (>= threshold) and "bad" (< threshold).
        """
        print("Classifying Users into Good and Bad...")
        self.data['Creditworthiness'] = np.where(self.data[rfms_score_col] >= threshold, 'Good', 'Bad')
        print(self.data[['RFMS_Score', 'Creditworthiness']].head())


