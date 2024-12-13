# Hospital Overcrowding Prediction with LIME

This project uses a machine learning model to predict hospital overcrowding based on patient diagnosis data and provides explanations for the predictions using LIME (Local Interpretable Model-agnostic Explanations) to make the model's decisions transparent and understandable.

## Overview

This application uses a Gradient Boosting Machine (GBM) model to predict hospital overcrowding from various patient-related data. It also uses LIME to provide interpretable explanations of the predictions, helping users understand the factors influencing the model's decisions.

## Data

The data used for this project was obtained from NHS Digital, specifically from the **Hospital Episode Statistics (HES)** dataset for admitted patient care during the 2023-24 financial year. The dataset was downloaded directly from the official NHS website [22], which provides comprehensive information about hospital admissions, diagnoses, treatments, and patient demographics.

## Features

- **Overcrowding Prediction:** Predict whether a hospital will experience overcrowding based on patient data.
- **What-If Analysis:** Allows users to select diagnosis categories and different features to simulate scenarios of hospital overcrowding.
- **LIME Explanations:** Provides an understandable explanation of each prediction using LIME, helping users see which features influenced the model's decision.

## Accessing the Dashboard

For online usage, you can access the hospital overcrowding prediction dashboard via the following link:

[Hospital Overcrowding Prediction Dashboard](https://hospital-overcrowding-lime-dashboard.streamlit.app/)

## How to Use the Application

1. **Select Diagnosis Category:** Choose a diagnosis category and select specific diagnoses to simulate hospital overcrowding scenarios.
2. **Input Feature Data:** Enter values for different features such as "waiting time," "emergency treatment," etc.
3. **Predict:** Click the "Predict" button to see the predicted overcrowding status of the hospital and the confidence level of the prediction.
4. **LIME Explanation:** After prediction, an explanation will show which features were most influential in the model's decision.

## Model Performance

The model was evaluated using **Accuracy**, **Precision**, **Recall**, and **F1 Score**, with the following results:

- **Accuracy:** 0.967
- **Precision:** 0.9444
- **Recall:** 0.8854
- **F1 Score:** 0.914

## Model Details

- **Model Type:** Gradient Boosting Machine (GBM)
- **Explanation Tool:** LIME (Local Interpretable Model-agnostic Explanations)
- **Training Data:** Data on patient admissions, diagnoses, and hospital activities.
