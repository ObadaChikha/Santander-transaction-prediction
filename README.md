# Santander Customer Transaction Prediction

## Overview
An end-to-end, hypothesis-driven machine learning pipeline developed to predict customer transactions on the anonymized, high-dimensional, and class-imbalanced Santander dataset (200,000 rows, 200 continuous features, ~90:10 target split).

📄 **Full Project Documentation & Logbook:** See [`Santander_Transaction_Prediction_Report.pdf`](./Santander_Transaction_Prediction_Report.pdf) for detailed methodology, statistical diagnostics, and iteration logs.

## Key Engineering & Analytical Highlights
* **Memory Optimization:** Reduced dataset memory footprint by over **50%** (from >300 MB down to 152.8 MB) without information loss by downcasting `float64` to `float32` and `int64` to `int8`.
* **Statistical EDA:** Verified Gaussian feature distributions and minimal pairwise correlation (max correlation ~`0.0809`), proving orthogonal feature independence and guiding the decision against aggressive outlier clipping or manual feature elimination.
* **Validation Strategy:** Implemented a strict **Stratified 5-Fold Cross-Validation** scheme using **ROC-AUC** as the primary metric to prevent data leakage and account for severe class imbalance.

## Model Progression & Results (5-Fold Mean Test ROC-AUC)

| Model / Pipeline | Test ROC-AUC | Notes |
| :--- | :---: | :--- |
| **Logistic Regression** (Baseline) | `0.8595` | Linear benchmark with `StandardScaler` |
| **Gaussian Naive Bayes** (Raw) | `0.8805` | Exploits orthogonal feature independence |
| **XGBoost** (Tuned) | `0.8836` | Level-wise growth (`max_depth=4`, `subsample=0.8`) |
| **Quantile Transformer + GaussianNB** | `0.8879` | Maps features to exact normal distributions |
| **LightGBM** (Tuned) | `0.8881` | Leaf-wise growth (`num_leaves=63`, `min_child_samples=50`) |
| **Final Heterogeneous Ensemble** | **`0.8946`** | **Soft Voting (`0.35` Q-GNB + `0.25` XGB + `0.40` LGBM)** |

## Tech Stack
* **Python:** `Pandas`, `NumPy`, `Scikit-Learn`, `XGBoost`, `LightGBM`, `Matplotlib`, `Seaborn`
