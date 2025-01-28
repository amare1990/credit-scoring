# Credit Scroring

> Credit Scoring is a data science project aimed at developing a robust Credit Scoring Model to evaluate the creditworthiness of potential borrowers. The model leverages data provided by the eCommerce platform and utilizes Python along with a suite of powerful libraries for data analysis, feature engineering, and predictive modeling.

## Built With

- Major languages used: Python3
- Libraries: numpy, pandas, matplotlib.pyplot, scikit-learn
- Tools and Technlogies used: jupyter notebook, Google Colab, Git, GitHub, Gitflow, VS code editor.

## Demonstration and Website

[Deployment link]()

## Getting Started

You can clone my project and use it freely, then contribute to this project.

- Get the local copy, by running `git clone https://github.com/amare1990/credit-scoring.git` command in the directory of your local machine.
- Go to the repo main directory, run `cd credit-scoring` command
- Create python environment by running `python3 -m venv venm-name`, where `ven-name` is your python environment you create
- Activate it by running:
   - `source venv-name/bin/activate` on linux os command prompt if you use linux os
   - `myenv\Scripts\activate` on windows os command prompt if you use windows os.

- After that you have to install all the necessary Python libraries and tools by running `pip install -r requirements.txt`
- To automate the workflow and execute ata time, run the `python main/main.py` from the root directory of the repo or can be used for running the entire pipeline end-to-end without manual intervention.
- To run this project and to experiment with individual components of the workflow, open `jupyter notebook` command from the main directory of the repo and run it.

### Prerequisites

- You have to install Python (version 3.8.10 minimum), pip, git, vscode.

### Dataset

- `data.csv` is the datset our workflows work on. The dataset has 95662 number of Rows and 16 number of Columns.

### Project Requirements
- Git, GitHub setup, adding `pylint' in the GitHub workflows
- Statistical and EDA analysis on the data, ploting
- Feature Engineering


#### GitHub Action and Following Python Coding Styles
- The `pylint` linters are added in the `.github/workflows` direcory of the repo.
- Make it to check when Pull request is created
- Run `pylint scripts/exploratory_data_analysis.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive --aggressive scripts/exploratory_data_analysis.py` to automatically fix some linters errors
- Run `pylint scripts/feature_engineering.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive --aggressive scripts/scripts/feature_engineering.py` to automatically fix some linters errors
- Run `pylint scripts/creadit_scoring.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive --aggressive scripts/scripts/credit_scoring.py` to automatically fix some linters errors
- Run `pylint scripts/model_building.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive --aggressive scripts/scripts/model_building.py` to automatically fix some linters errors


### Exploratory Data Analysis

This part of this project conducts Exploraory Data Analysis (EDA) on the dataset to explore the dataset. The functionality is implemented in `exploratory_data_analysis.py` module.
In this portion of the task, the following analysis has been conducted.

- Overviewing data:
    Number of rows and columns, and their datatypes, printing the first five samples.
- Data Summary:
    Summarize statistical descriptive statistics for both numerical features and object type features too.

- Visualize the distribution of numerical features to identify patterns, skewness, and outliers.
- Visualize and analyze the distribution of categorical features.
- Visualize and analyze the correlation between numerical features
- Identify missing values in the dataset.
- Use box plots to identify outliers in numerical features.

### Feature Engineering
- The functionality is implemented in `feature_engineering.py` module.
- Create aggregate features such as sum, mean, count, and standard deviation for a numerical variable.
- Extract date/time-related features from a TransactionStartTime column.
- Encode categorical variables using One-Hot Encoding or Label Encoding based on users' preference.
- Handle missing values in the dataset using imputation or removal.
- Handle outliers in numerical columns using the specified method.
- Normalize or standardize numerical features.
- Save the processed dataset to a CSV file.

### Credit Scoring

- It is implemented in `credit_scoring.py` script.
- Each user's RFMS score with its corresponding CustomerId is computed and classified as `Bad` or `Good`.
- Calculate the RFMS score for each user
- Classify users as "good" or "bad" based on their RFMS score
- Visualize the RFMS score distribution and the threshold boundary.
- Perform Weight of Evidence (WoE) binning for the target variable.
- Visualize the WoE values of a feature.

### Model Training

- It is implemented in `model_building.py` script.
- Split data into training and testing data
- Four models: Logistic regression, Decision tree, RandomForest, and Gradient Boosting models are trained and evaluated
- Hyper-parameter tuning for RandomForest is conducted
- The ROC curve for all models is done
- Model performance results are displayed
- The models built are saved using pickle package

# Model Serving API
- Created ml_api django project
- Created predictions django app
- Implemented functionalities for serving saved model apis and able to predict using django restframework


> #### You can gain more insights by running the jupter notebook and view plots.


### More information
- You can refer to [this link]() to gain more insights about the reports of this project results.

## Authors

👤 **Amare Kassa**

- GitHub: [@githubhandle](https://github.com/amare1990)
- Twitter: [@twitterhandle](https://twitter.com/@amaremek)
- LinkedIn: [@linkedInHandle](https://www.linkedin.com/in/amaremek/)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](https://github.com/amare1990/credit-scoring/issues).

## Show your support

Give a ⭐️ if you like this project, and you are welcome to contribute to this project!

## Acknowledgments

- Hat tip to anyone whose code was referenced to.
- Thanks to the 10 academy and Kifiya financial instituion that gives me an opportunity to do this project

## 📝 License

This project is [MIT](./LICENSE) licensed.
