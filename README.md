# Flight Price Prediction Web Application

**Dataset**: https://www.kaggle.com/datasets/aathavanap/flight-ticket-price-prediction  
**Web Application**: https://pritam-flightpricespredictor.streamlit.app/

## Overview
This project involves the development of a web application that predicts flight prices using machine learning models. The dataset was sourced from Kaggle, and the entire process—from data cleaning to model deployment—was executed using Python, with a focus on functional programming techniques.

## Table of Contents
1. [Data Cleaning](#data-cleaning)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Feature Engineering](#feature-engineering)
4. [Model Training and Selection](#model-training-and-selection)
5. [Results](#results)
6. [How to Run the Application](#how-to-run-the-application)

## Data Cleaning
The data cleaning process involved several key steps to ensure the dataset's integrity:

- **Data Type Correction**: Fixed data types of columns to appropriate formats.
- **Duplicate Removal**: Identified and removed duplicate entries.
- **Error Correction**: Addressed inconsistencies, such as different spellings for the same value.
- **Column Value Conversion**: Converted column values to more useful formats, e.g., converting flight duration into minutes.
- **Data Splitting**: Split the cleaned data into training and testing sets to prevent data leakage.

Functional programming techniques were utilized throughout this process for better code organization and reusability.

## Exploratory Data Analysis (EDA)
EDA was performed exclusively on the training data to avoid data leakage, focusing on both univariate and multivariate analyses:

- **CRAMER’S V Test**: Used to determine the correlation between categorical variables.
- **ANOVA Test**: Assessed the association between categorical features and the numerical target variable.
- **SHAPIRO-WILK Test**: Checked for normality in numerical columns.
- **SPEARMAN Test**: Analyzed correlations between numerical features and the target variable.

Given that tree-based models were used later in the process, multicollinearity was not assessed.

## Feature Engineering
Feature engineering was a critical component of the project, conducted using pipelines for streamlined processing:

- **Custom Feature Extraction**: Developed custom functions to derive new features from existing data.
- **Categorical Encoding**: Applied techniques like one-hot encoding, ordinal encoding, and grouped rare categories using `RareLabelEncoder` from the `Feature-Engine` library.
- **Outlier Removal**: Used the `WINSORIZER` method to handle outliers.
- **Feature Scaling**: Applied `StandardScaler` and `MinMaxScaler` for scaling numerical features.
- **Feature Selection**: Leveraged `SelectBySingleFeaturePerformance` with `RandomForestRegressor` as the estimator, reducing the feature set from 32 to 16.

## Model Training and Selection
Two machine learning models were trained and tuned for flight price prediction:

- **RandomForestRegressor**
- **XGBoostRegressor**

Both models were tuned using `GridSearchCV` to identify optimal hyperparameters. The following metrics were considered:

- **R² Score**
- **Mean Absolute Error (MAE)**

Based on performance, XGBoostRegressor was selected for deployment due to its superior R² score of 75% and MAE of 1435.

## Results
- **RandomForestRegressor**:
  - **R² Score**: 70%
  - **MAE**: 1726
- **XGBoostRegressor**:
  - **R² Score**: 75%
  - **MAE**: 1435

## How to Run the Application
To run the application locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/PritamBanerjee21/Flight_Prices_Prediction_Project/tree/main
    ```
2. Navigate to the project directory:
    ```bash
    cd flight-price-prediction
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
    streamlit run app.py
    ```

The web application should now be accessible via your local browser.

---

Feel free to adjust any parts as necessary before uploading it to GitHub!
