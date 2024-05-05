# Electrical Circuit Recognition Project - BSDSA

## Overview
This repository hosts the project I led, focusing on the recognition of electrical circuits. The project involves the use of a Regional Convolutional Neural Network (R-CNN), specifically a Faster R-CNN with a ResNet50 backbone, optimized for the classification and localization of electrical components in circuit images.

## Project Structure
The repository consists of a comprehensive Jupyter notebook that covers all aspects of the project from data preparation to model evaluation:

### CircuitRecognition.ipynb
- **Purpose**: Handling the complete lifecycle of the CNN model for recognizing electrical circuits.
- **Key Tasks**:
  - Preparing and preprocessing image data and corresponding annotations.
  - Building and training the Faster R-CNN model.
  - Implementing GPU acceleration support and dynamic memory management.
  - Evaluating the model using metrics like precision, recall, F1-score, and IoU (Intersection over Union).
  - Outputting a detailed performance report and saving the trained model.

## Libraries Used
- PyTorch
- Pandas
- NumPy
- PIL (Python Imaging Library)
- Torchvision

## How to Run on Kaggle
1. Log in to your Kaggle account or create one if you don't have it.
2. Upload the notebook `CircuitRecognition.ipynb` to your Kaggle workspace.
3. Make sure to select the dataset within Kaggle.
4. Execute the notebook from start to finish to reproduce the training and evaluation process.
5. Use the GPU acceleration provided by Kaggle for optimal performance.

## Additional Notes
- Ensure you have appropriate hardware (preferably with CUDA support) to handle model training if running locally.
- The dataset paths in the notebook are configured for execution on Kaggle. Adjust these paths if running in a different environment.

## Project Members
- Giorgio Micaletto
- Denis Magzhanov
- Dmitry Kuznetsov
- Demir Elmas
- Andrei Sofronie
