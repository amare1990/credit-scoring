import pandas as pd


from scripts.explotary_data_analysis import ExploratoryDataAnalysis
from scripts.feature_engineering import FeatureEngineering
from scripts.credit_scoring import CreditScoring
from scripts.model_building import ModelPipeline


if __name__ == "__main__":

    """
    Exploratory Data Analysis
    """

    # Instantiate the exploratory data analysis
    data_path = '../data/data.csv'
    eda = ExploratoryDataAnalysis(data_path)

    # Run methods
    eda.overview_of_data()
    eda.summary_statistics()
    eda.distribution_of_numerical_features()
    print('Visualization of numerical features completed!')
    eda.distribution_of_categorical_features()
    print('Visualization of categorical features completed!')
    eda.correlation_analysis()
    print('Heatmap construction of numerical features completed!')
    eda.identify_missing_values()
    print('Identifying missing values completed!')
    eda.outlier_detection()
    print('Identifying outliers completed!')

    """
    Feature Engineering
    """
    # Run feature engineering methods

    data_path = '../data/data.csv'
    feature_eng = FeatureEngineering(data_path=data_path)

    print('\n\n*************************************************************\n\n')

    # Apply feature engineering methods
    feature_eng.create_aggregate_features()

    print('\n\n*************************************************************\n\n')
    feature_eng.calculate_recency()
    if 'Recency' in feature_eng.data.columns:
        print("'Recency' column created successfully.")
        print(feature_eng.data['Recency'].isna().sum())
    else:
        print("'Recency' column not found. Check calculate_recency method.")
    print('\n\n*************************************************************\n\n')
    feature_eng.extract_features()

    print('\n\n*************************************************************\n\n')
    feature_eng.handle_missing_values(strategy="mean")  # Default for numerical
    print('\n\n*************************************************************\n\n')
    feature_eng.handle_missing_values(
    strategy="most_frequent")  # For categorical
    print('\n\n*************************************************************\n\n')
    feature_eng.handle_outliers(method="iqr", factor=3)

    print('\n\n*************************************************************\n\n')
    feature_eng.encode_categorical_variables(method="one_hot")

    print('\n\n*************************************************************\n\n')
    feature_eng.normalize_or_standardize(method="standardize")

    print('\n\n*************************************************************\n\n')

    # Save processed data to a temporary file.
    output_path = "../data/feature_engineered_data.csv"
    feature_eng.save_processed_data(output_path)

    """
    Running credit scoring methods
    """
    print('\n\n*************************************************************\n\n')
    data_path = '../data/feature_engineered_data.csv'
    cs = CreditScoring(data_path)

    cs.calculate_rfms(
    recency_col='Recency',
    frequency_col='Transaction_Count',
    monetary_col='Total_Transaction_Amount'
    )

    cs.classify_users(rfms_score_col='RFMS_Score', threshold=0.5)

    # Save credit scored data
    output_path = '../data/credit_scored_data.csv'
    cs.save_credit_scored_data(output_path)

    # WoE binning without global `exclude_columns` variable
    # Apply WoE binning
    bins_adj = cs.apply_woe_binning_monotonic(target_col='Creditworthiness',
                                        exclude_cols=['TransactionId', 'BatchId',
                                        'SubscriptionId', 'AccountId', 'CustomerId', 'MostRecentTransaction', 'CountryCode'])
    # Visualization step
    feature_to_visualize = 'Recency'
    if feature_to_visualize in bins_adj:
        cs.visualize_woe_binning(bins_adj, feature_to_visualize)
    else:
        print(
            f"Feature '{feature_to_visualize}' not found in the bins dictionary.")



    """
    Run model building methods
    """
    import re

    data = '../data/credit_scored_data.csv'
    df = pd.read_csv(data)

    # Define the base column names and patterns for regex matching
    base_cols = ['RFMS_Score', 'FraudResult', 'Creditworthiness']
    regex_patterns = ['ChannelId_.*', 'ProductId_.*', 'ProductCategory_.*', 'ProviderId_.*', 'transaction_.*']

    # Use regex to find matching columns
    matching_cols = []
    for pattern in regex_patterns:
        matching_cols.extend([col for col in df.columns if re.match(pattern, col)])

    # Combine base columns with dynamically matched columns
    include_cols = base_cols + matching_cols

    # Filter the DataFrame
    df = df[include_cols]

    # Print the final column names
    print(df.columns)

    target_col = 'Creditworthiness'
    data = df

    model_pipeline = ModelPipeline(data, target_col)
    model_pipeline.split_data()
    model_pipeline.train_models()
    model_pipeline.hyperparameter_tuning()
    model_pipeline.evaluate_models()
    model_pipeline.visualize_roc_curve()
    model_pipeline.display_results()
    model_pipeline.save_models()
