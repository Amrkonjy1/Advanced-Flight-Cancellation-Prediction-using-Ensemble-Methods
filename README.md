# Advanced Flight Cancellation Prediction using Ensemble Methods ✈️

## 📖 Overview

This project presents an in-depth analysis of the **2015 US Flight Delays and Cancellations** dataset to predict the likelihood of a flight being canceled. This repository showcases a complete machine learning workflow, from data cleaning and feature engineering to model training and evaluation using powerful ensemble methods like **Random Forest** and **XGBoost**.

The core objective is to build a robust classification model capable of forecasting flight cancellations with high accuracy. The analysis also demonstrates effective techniques for handling a highly imbalanced dataset with **SMOTE** and encoding high-cardinality categorical features, such as airport and airline codes, which are common challenges in real-world data.

---

## 🛠️ Methodology

The project follows a structured machine learning pipeline to ensure robust and reproducible results.

### 1. **Data Preprocessing and Feature Engineering**
-   **Feature Aggregation:** To simplify the feature set, the five distinct delay-related columns were consolidated into a single `total_delay_time` feature. This reduces dimensionality while retaining crucial information about flight delays.
-   **Handling Missing Values:** Missing data points in numerical columns were imputed using the mean of their respective features. The `CANCELLATION_REASON` column was removed to prevent data leakage, as it is only present for flights that have already been canceled.
-   **Categorical Encoding:** All categorical features, including `AIRLINE`, `TAIL_NUMBER`, `ORIGIN_AIRPORT`, and `DESTINATION_AIRPORT`, were transformed into a numerical format using scikit-learn's `OrdinalEncoder`. The encoder for the `AIRLINE` feature was specifically configured to handle new, unseen categories in test data by assigning them a predefined value, preventing errors during inference.

### 2. **Handling Class Imbalance with SMOTE**
-   The dataset exhibited a significant class imbalance, with a very small percentage of flights being canceled compared to those that flew successfully.
-   To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training set. This technique generates synthetic data points for the minority class (canceled flights), creating a balanced dataset that prevents the model from being biased towards the majority class.

### 3. **Model Training and Evaluation**
-   The balanced dataset was split into training (80%) and testing (20%) sets. All features were scaled using `StandardScaler` to normalize their range.
-   Two powerful, tree-based **ensemble models** were trained:
    -   **Random Forest Classifier**
    -   **XGBoost Classifier**
-   Model performance was rigorously evaluated using accuracy scores and confusion matrices to assess their ability to correctly classify both canceled and successful flights.

---

## 📊 Results & Key Findings

The models achieved strong predictive performance on the unseen test data, demonstrating the effectiveness of the methodology.

-   **Random Forest:** This model emerged as the top performer, achieving an impressive accuracy of **98.65%**. Feature importance analysis revealed that `total_delay_time` and `TAIL_NUMBER` were among the most influential predictors of a cancellation.
-   **XGBoost:** The XGBoost model also delivered a solid performance with an accuracy of **92.31%**, serving as a strong alternative.

The high accuracy scores and detailed confusion matrices confirm that the trained models are highly capable of distinguishing between flights that are likely to be canceled and those that are not. The complete analysis, code, and visualizations are available within the Jupyter Notebook.
