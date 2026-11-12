# Santander Customer Transaction Prediction

## Overview
An end-to-end predictive modeling pipeline developed to identify customer transaction patterns on an anonymized, high-dimensional, and class-imbalanced financial dataset.

## Methodology & Tech Stack
* **Language & Libraries:** Python (`Pandas`, `NumPy`, `Scikit-Learn`, `XGBoost`, `LightGBM`, `Matplotlib`, `Seaborn`)
* **Exploratory Data Analysis (EDA):** Analyzed feature distributions, correlation structures, and class imbalance across 200 anonymized numerical features.
* **Modeling & Evaluation:** Trained and benchmarked multiple classification algorithms including **LightGBM**, **XGBoost**, **Random Forest**, and **Gaussian Naive Bayes**, evaluating performance primarily using **ROC-AUC** to account for class imbalance.

## Key Takeaways
* Gradient boosting frameworks (**LightGBM** and **XGBoost**) significantly outperformed baseline models in capturing subtle non-linear patterns across uncorrelated features.
* Proper hyperparameter tuning and cross-validation prevented overfitting on high-dimensional tabular data.
