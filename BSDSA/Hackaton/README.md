# BDSDA Hackaton

Welcome to the Hackaton folder. This directory contains Python scripts related to a hackathon competition and hyperparameter optimization for a machine learning model. Below, you'll find a brief description of each script and its purpose.

## main.py

### Overview
- **Description**: This script focuses on a machine learning hackathon competition.
- **Objective**: The goal of the hackathon was to build a predictive model to solve a specific problem. The winning team was determined based on the production of the best lift curve from their predictions.
- **Data**: It utilizes training data from 'train.csv' and makes predictions on test data from 'test_no_tgt.csv.'
- **Key Steps**:
  1. Data preprocessing and feature engineering.
  2. Encoding categorical features.
  3. Training an XGBoost classifier with predefined hyperparameters.
  4. Assessing the model's performance using ROC AUC score.
  5. Generating predictions for the test data and saving them in 'submission.csv.'

## Hyperparameter_optimization.py

### Overview
- **Description**: This script focuses on hyperparameter optimization for an XGBoost classifier.
- **Objective**: It aims to find the best set of hyperparameters for the XGBoost model to improve its performance in the hackathon competition.
- **Data**: Similar to "main.py," it uses data from 'train.csv' and 'test_no_tgt.csv', which unfortunately I can't share.
- **Key Steps**:
  1. Data preprocessing and feature engineering.
  2. Encoding categorical features.
  3. Implementing hyperparameter optimization using Optuna, specifically the TPE sampler.
  4. Finding the best hyperparameters that minimize the difference between the training and testing ROC AUC.
  5. Training the XGBoost classifier with the optimized hyperparameters.
  6. Evaluating the model's performance using ROC AUC score on the test set.

## Team Achievement

I'm pleased to announce that the team to which I contributed in the hackathon won. The winning team was determined based on the production of the best lift curve from their predictions. This victory reflects the dedication and collaborative efforts of our team members.

If you have any questions or inquiries, please don't hesitate to reach out.
