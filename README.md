### Streamlit App (click the button below!)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection.streamlit.app/)

# Credit Card Fraud Detection

A machine learning model using advanced feature engineering and decision tree classification to accurately detect fraudulent credit card transactions.

## Key Improvements

- **Enhanced Model Performance**: F1 score improved from 0.64 to 0.99
- **Advanced Feature Engineering**: Added statistical anomaly scores (Z-scores) and balance change features
- **Robust Class Imbalance Handling**: Used class weight balancing and stratified sampling
- **Overfitting Prevention**: Limited tree depth and used cross-validation
- **Local Model Inference**: No external API dependency, all prediction happens locally

## What is Credit card fraud?

Credit card fraud is a form of identity theft that involves an unauthorized taking of another's credit card information for the purpose of charging purchases to the account or removing funds from it.

## Introduction

In this project, we developed a machine learning model using classification algorithms and advanced feature engineering techniques to accurately detect if a credit card transaction is fraudulent or not.

## Data Sourcing
 
Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1
 
A synthetic dataset generated using the simulator called PaySim was used as the dataset for building the model used in this project. PaySim uses aggregated data from the private dataset to generate a synthetic dataset that resembles the normal operation of transactions and injects malicious behaviour to later evaluate the performance of fraud detection methods.

## App Interface

![Home page](home_credit.gif)
