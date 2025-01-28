"""
Model training
"""

import os
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


class ModelPipeline:
    def __init__(self, data, target_col):
        """
        Initialize the ModelPipeline class with the dataset and target variable.
        :param data: Pandas DataFrame containing the dataset.
        :param target_col: Name of the target column in the dataset.
        """
        self.data = data
        self.target_col = target_col
        self.X = data.drop(columns=[target_col])
        self.y = data[target_col]
        self.models = {}
        self.results = {}


    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        :param test_size: Proportion of the data to use for testing.
        :param random_state: Random state for reproducibility.
        """
        print("Splitting data into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Training set size: {self.X_train.shape[0]}, Testing set size: {self.X_test.shape[0]}")


    def train_models(self):
        """
        Train Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models.
        """
        print("Training models...")
        self.models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        self.models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
        self.models['Random Forest'] = RandomForestClassifier(random_state=42)
        self.models['Gradient Boosting'] = GradientBoostingClassifier(random_state=42)

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(self.X_train, self.y_train)

    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning using GridSearchCV for Random Forest.
        """
        print("Performing hyperparameter tuning for Random Forest...")
        rf_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
        }
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        self.models['Tuned Random Forest'] = grid_search.best_estimator_
        print(f"Best parameters for Random Forest: {grid_search.best_params_}")


    def evaluate_models(self):
        """
        Evaluate models using accuracy, precision, recall, F1 score, and ROC-AUC.
        """
        print("Evaluating models...")
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='binary')
            recall = recall_score(self.y_test, y_pred, average='binary')
            f1 = f1_score(self.y_test, y_pred, average='binary')
            roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None

            self.results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC-AUC': roc_auc,
            }

        # Loop through results correctly
        for model_name, metrics in self.results.items():
            print(f'Results for {model_name}')
            for metric, value in metrics.items():
                print(f'{metric}: {value}')





    def save_models(self, directory='../data/saved_models'):
        """
        Save trained models to disk using pickle.
        :param directory: Directory to save the models.
        """
        print(f"Saving models to '{directory}'...")
        os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
        for model_name, model in self.models.items():
            filepath = os.path.join(directory, f'{model_name.replace(" ", "_")}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)  # Save the model using pickle
            print(f"{model_name} saved at {filepath}")



    def visualize_roc_curve(self):
        """
        Plot the ROC curve for all models.
        """
        print("Visualizing ROC curves...")
        plt.figure(figsize=(10, 6))
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(self.y_test, y_pred_proba):.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.show()


    def display_results(self):
        """
        Display evaluation metrics for all models.
        """
        print("Model Evaluation Results:")
        results_df = pd.DataFrame(self.results).T
        print(results_df)



