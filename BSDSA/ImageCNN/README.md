# Image Classification CNN Project - BSDSA

## Overview
This repository contains the work I completed as part of a project with Bocconi Statistics and Data Science students Association (BSDSA). The project focuses on building, optimizing, and explaining a Convolutional Neural Network (CNN) for image classification.

## Project Structure
The repository consists of two main notebooks:

### 1. Hyperparameters.ipynb
- **Purpose**: Building the CNN model and optimizing its hyperparameters.
- **Key Tasks**:
  - Designing the CNN architecture.
  - Implementing optuna for hyperparameter tuning to enhance model performance.
  - Write a custom EarlyStopping class to avoid overfitting

### 2. FeatureMaps.ipynb
- **Purpose**: Training the CNN model and extracting feature maps for the first and last layers.
- **Key Tasks**:
  - Training the model with optimized hyperparameters.
  - Visualizing feature maps to understand model decisions and enhance explainability.

## Libraries Used
- TensorFlow
- Optuna
- Sci-kit Learn
- Kaggle

## Usage
To explore these notebooks, you can clone the repository and run the Jupyter notebooks in an environment that supports TensorFlow and Keras.

```bash
git clone https://github.com/GiorgioMB/UniversityProjects.git
cd UniversityProjects/BSDSA/ImageCNN```
