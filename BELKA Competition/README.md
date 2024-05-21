# BELKA Competition Submission

Welcome to my submission for the BELKA Competition! This repository contains the code and models developed to predict the binding affinity of small molecule drugs to specified protein targets. 

## Introduction
Small molecule drugs interact with cellular protein machinery to affect their functions. Traditionally, identifying candidate molecules involves physically testing them, which is labor-intensive and 
time-consuming. The FDA has approved approximately 2,000 novel molecular entities, yet the chemical space of druglike molecules is estimated to be around 10^60, making physical searches impractical.
Leash Biosciences provided a dataset, the Big Encoded Library for Chemical Assessment (BELKA), containing data on 133M small molecules tested for their interaction with three protein targets using DNA-encoded 
chemical library (DEL) technology. 

## Files
The repository consists of the following files:
* `train.py`: This script contains the code used for training the predictive models. It includes data preprocessing, model configuration, training, and evaluation steps.
* `test.py`: This script contains the trained models and code for testing the binding affinity predictions on new chemical compounds.

## Usage
### Training

To train the predictive models, follow these steps:
1. Ensure you have the training dataset (`train.parquet`) in the appropriate directory.
2. Run the `train.py` script to start the training process.
    ```bash
    python3 train.py
    ```
   
The script will load the data, preprocess it, and train the models. The trained model will be saved as `model.pth`.

### Testing
To use the trained predictive models, follow these steps:
1. Ensure you have the test dataset (`test.parquet`) in the appropriate directory.
2. Run the `test.py` script to start the testing process.
    ```bash
    python3 test.py
    ```

The script will load the trained models, process the test data, and output the predictions to `predictions.csv`.

## Code Overview

### train.py

This script handles the training process. Key components include:

- **Data Preprocessing**: Converting SMILES strings to molecular graphs, using Dask and RDKit for conversion.
- **Model Definition**: The `MultiModelGNNBind` class, a stacked meta-model combining molecular fingerprinting and graph isomorphism network for binding affinity prediction.
- **Training Loop**: The `train_model` function, which trains the model over a specified number of epochs.

### test.py

This script handles the testing and prediction process. Key components include:

- **Data Loading**: Loading the test dataset and preprocessing it similarly to the training data.
- **Model Loading**: Loading the trained model from `model.pth`.
- **Prediction Loop**: Making predictions on the test data and saving the results to `predictions.csv`.

## Dependencies

Ensure you have the following dependencies installed:
- pandas
- numpy
- torch
- rdkit
- bioservices
- dask
- scikit-learn
- torch-geometric
- psutil
- joblib
- duckdb

You can install the required packages using pip:
```bash
pip install pandas numpy torch rdkit bioservices dask scikit-learn torch-geometric psutil joblib duckdb
```
## Conclusion
Feel free to explore, learn, and adapt this project to your needs. If you have any questions or suggestions, please feel free to reach out.
