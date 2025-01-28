"""
Model training
"""

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
