import pandas as pd


from scripts.explotary_data_analysis import ExploratoryDataAnalysis
from scripts.feature_engineering import FeatureEngineering



# Exploratory Data Analysis
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
  # Processing data in chunks
  chunk_size = 10000  # Define a manageable chunk size
  chunks = pd.read_csv(data_path, chunksize=chunk_size)

  # Loop through each chunk
  for i, chunk in enumerate(chunks):
      print(f"Processing chunk {i+1}")

      # Pass the chunk directly to the FeatureEngineering class
      feature_eng = FeatureEngineering(chunk)  # No 'data=' keyword

      # Apply feature engineering methods
      feature_eng.create_aggregate_features()
      feature_eng.extract_features()
      feature_eng.encode_categorical_variables(method="one_hot")
      feature_eng.handle_missing_values(strategy="mean")
      feature_eng.handle_outliers(method="iqr")
      feature_eng.normalize_or_standardize(method="standardize")

      # Save each processed chunk to a temporary file or append to a combined dataset
      chunk_output_path = f"../data/processed_chunk_{i+1}.csv"
      feature_eng.save_processed_data(chunk_output_path)


  """
  Combining each chunked data.
  """
  import glob

  # List all processed chunk files
  processed_files = glob.glob("../data/processed_chunk_*.csv")

  # Combine them into a single DataFrame
  combined_data = pd.concat([pd.read_csv(file) for file in processed_files])

  # Save the combined data
  combined_data.to_csv('../data/processed_data_combined.csv', index=False)



