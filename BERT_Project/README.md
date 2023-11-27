# BERT Project
Welcome to the Financial Sentiment Analysis project using BERT!  This repository contains the code and model necessary to analyze sentiment in finance-related textual data such as news headlines, tweets, and short phrases.  This project represents my introduction to natural language processing (NLP) and machine learning.

## Introduction
In this project, I have developed a sentiment analysis model using BERT to determine the sentiment (positive, negative, or neutral) of financial-related statements. The model has been trained on a dataset containing news headlines, tweets, and short phrases related to finance. The goal is to provide a tool that can help analyze the sentiment behind financial text data, which can be useful for various applications such as market sentiment analysis, trend prediction, and more.
The dataset used to train this model can be found [here](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

## Files
The repository consists of the following files:
* Financial Sentiment Analysis.ipynb: This Jupyter Notebook contains the trained BERT model that can be used to analyze the sentiment of financial text inputs provided by the user.
* BERT Training.ipynb: This Jupyter Notebook contains the code used for training the BERT model. It includes data preprocessing, model configuration, training, and evaluation steps.

## Usage
To use the trained sentiment analysis model, follow these steps:
* Open the Financial Sentiment Analysis.ipynb notebook.
* Follow the outlined steps to load the trained BERT model.
* Run the notebook cells to initialize the model.
* Input a financial-related statement (news headline, tweet, or short phrase) in the designated section.
* The model will analyze the sentiment of the input and provide an output indicating whether the sentiment is positive, negative, or neutral, as well as the model's confidence score (ranging from 0 to 1) of its prediction.

For those interested in the training process, the BERT Training.ipynb notebook provides insights into how the model was trained. It covers data preprocessing, model configuration using Hugging Face's Transformers library, fine-tuning, and evaluation.

---
Feel free to explore, learn, and adapt this project to your needs. 

If you have any questions or suggestions, please feel free to reach out.
