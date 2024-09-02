# House Price Prediction Project

This project aims to build a machine learning model to predict house prices using a dataset that contains various features related to houses in Ames, Iowa. The dataset is sourced from a Kaggle competition, where participants are required to predict the prices of different types of houses based on the features provided.

## Dataset Overview

- **Source**: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation)
- **Description**: The dataset consists of 79 explanatory variables that describe the characteristics of houses. These variables include both numerical and categorical features.
- **Train Set**: 1460 observations with target variable (SalePrice) available.
- **Test Set**: Contains similar observations but with the target variable missing, which needs to be predicted.

### Dataset Composition

- **Total Observations**: 1460 (Train Set)
- **Numerical Variables**: 38
- **Categorical Variables**: 43

## Problem Statement

The goal is to develop a machine learning model that can accurately predict the sale prices of houses based on the given features. The dataset includes various attributes of the houses, and the challenge is to minimize the error in price predictions.

# Installion
## 1.Clone the repository:

➜ git clone https://github.com/yourusername/house-price-prediction.git

## 2.Install the required dependencies:
➜ pip install -r requirements.txt


# How to Run
The main model used in this project is LightGBM, a gradient boosting framework that uses tree-based learning algorithms. The model is optimized using GridSearchCV to find the best combination of hyperparameters. After training the model with optimized parameters, the house prices in the test set are predicted and saved for submission.
To run the main script:

➜ python src/main.py
