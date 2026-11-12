# Santander Customer Transaction Prediction

## Overview
An end-to-end, hypothesis-driven machine learning pipeline developed to predict customer transactions on the anonymized, high-dimensional, and class-imbalanced Santander dataset (200,000 rows, 200 continuous features, ~90:10 target split)[cite: 10, 17].

📄 **Full Project Documentation & Logbook:** See [`ADA_133_Chikha_Portfolio.pdf`](./ADA_133_Chikha_Portfolio.pdf) for detailed methodology, statistical diagnostics, and iteration logs.

## Key Engineering & Analytical Highlights
* **Memory Optimization:** Reduced dataset memory footprint by over **50%** (from >300 MB down to 152.8 MB) without information loss by downcasting `float64` to `float32` and `int64` to `int8`[cite: 10, 17].
* **Statistical EDA:** Verified Gaussian feature distributions and minimal pairwise correlation (max correlation ~`0.0809`), proving orthogonal feature independence and guiding the decision against aggressive outlier clipping or manual feature elimination[cite: 11, 19, 20].
* **Validation Strategy:** Implemented a strict **Stratified 5-Fold Cross-Validation** scheme using **ROC-AUC** as the primary metric to prevent data leakage and account for severe class imbalance[cite: 10, 11, 18, 22].

## Model Progression & Results (5-Fold Mean Test ROC-AUC)

| Model / Pipeline | Test ROC-AUC | Notes |
| :--- | :---: | :--- |
| **Logistic Regression** (Baseline) | `0.8595` | Linear benchmark with `StandardScaler`[cite: 12, 20, 21] |
| **Gaussian Naive Bayes** (Raw) | `0.8805` | Exploits orthogonal feature independence[cite: 11, 12] |
| **XGBoost** (Tuned) | `0.8836` | Level-wise growth (`max_depth=4`, `subsample=0.8`)[cite: 11, 12] |
| **Quantile Transformer + GaussianNB** | `0.8879` | Maps features to exact normal distributions[cite: 11, 12] |
| **LightGBM** (Tuned) | `0.8881` | Leaf-wise growth (`num_leaves=63`, `min_child_samples=50`)[cite: 12] |
| **Final Heterogeneous Ensemble** | **`0.8946`** | **Soft Voting (`0.35` Q-GNB + `0.25` XGB + `0.40` LGBM)**[cite: 12, 13] |

## Tech Stack
* **Python:** `Pandas`, `NumPy`, `Scikit-Learn`, `XGBoost`, `LightGBM`, `Matplotlib`, `Seaborn`
