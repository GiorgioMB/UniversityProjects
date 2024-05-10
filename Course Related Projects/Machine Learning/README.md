# Project Overview
As the sole developer and team lead on this project, I focused on creating a comprehensive machine learning pipeline for credit scoring. 
This project involved stages from data preprocessing to the application of models such as Logistic Regression, Neural Networks, XGBoost, and Randomer Forest (SPORF). 
I personally updated and restored the SPORF library, ensuring its applicability for this and future projects.

# Project Structure
- Data Preprocessing: The Drive is mounted and the datasets are unzipped and loaded
- Data Exploration: Using AutoViz we checked for `NaN`s, correlation and further checked problematic datapoints
- Data Cleaning:
  - Correlation is removed by taking the correlated features as the residual of an OLS of the original features
  - `NaN`s are imputed with IterativeImputer using as the estimator a `RandomForest`
  - Synthetic datapoints are created for the minority class using SMOTE
- Model Fitting and evaluation: We fitted and evaluated several models, providing explanation with SHAP for the Logistic Regression and the XGBoost model

# Libraries Used
- `pandas`, `numpy`: Used for data manipulation and numerical computations.
- `matplotlib`, `seaborn`: For data visualization.
- `sklearn`: Provides tools for data preprocessing, model building, and evaluation.
- `imbalanced-learn`: For handling imbalanced data through resampling techniques.
- `shap`: For model interpretation and generating SHAP values.
- `optuna`: For optimizing model parameters.
- `torch`: Utilized for building and training neural network models.
- `AutoViz`: Automatically visualizes data to speed up the EDA process.
- `xgboost`: For training gradient boosting models.
