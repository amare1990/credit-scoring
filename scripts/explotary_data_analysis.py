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
        print("Summary Statistics for numerical features:")
        print(self.data.describe())
        print("\n\n")
        print("Summary Statistics for categorical features:")
        print(self.data.describe(include=[object, 'category']))
        print("\n")

    def retrieve_numerical_columns(self):
        """
        Extract numerical features from the dataset and return.
        """
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

        return numerical_columns

    def distribution_of_numerical_features(self):
        """
        Visualize the distribution of numerical features to identify patterns, skewness, and outliers.
        """
        print("Distribution of Numerical Features:")
        numerical_columns = self.retrieve_numerical_columns()
        for column in numerical_columns:
            sns.histplot(self.data[column], kde=True, bins=30)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.savefig(f"plots/numerical_features/{column}.png", dpi=300, bbox_inches='tight')
            plt.show()

    def distribution_of_categorical_features(self):
        """
        Analyze the distribution of categorical features.
        """
        print("Distribution of Categorical Features:")
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            sns.countplot(x=self.data[column])
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.savefig(f"plots/categorical_features/{column}.png", dpi=300, bbox_inches='tight')
            plt.show()

    def correlation_analysis(self):
        """
        Analyze the correlation between numerical features.
        """
        print("Correlation Analysis:")
        numerical_columns = self.retrieve_numerical_columns()
        corr_matrix = self.data[numerical_columns].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig(f"plots/heatmap/correlation.png", dpi=300, bbox_inches='tight')
        plt.show()

    def identify_missing_values(self):
        """
        Identify missing values in the dataset.
        """
        print("Missing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        print("\n")

    def outlier_detection(self):
        """
        Use box plots to identify outliers in numerical features.
        """
        print("Outlier Detection:")
        numerical_columns = self.retrieve_numerical_columns()
        for column in numerical_columns:
            sns.boxplot(x=self.data[column])
            plt.title(f"Boxplot of {column}")
            plt.xlabel(column)
            plt.show()
