# 🕵️ IEEE-CIS Fraud Detection: Machine Learning Pipeline

This repository contains a complete machine learning pipeline for detecting fraudulent transactions using the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection). The goal is to build a production-grade fraud detection model, with feature engineering, class imbalance handling, and interpretable model evaluation.

---

## 📊 Dataset Description

The dataset includes anonymized transactional and identity data. Key challenges:
- Highly imbalanced classes (~3.6% fraud)
- Many categorical and masked features
- Rich digital fingerprints (`DeviceInfo`, `id_*`)
- Obfuscated engineered features (`V1–V339`)

---

## 🧠 Goals of the Project

- Extract meaningful signals from messy, sparse fraud data
- Handle extreme class imbalance using SMOTE
- Build a high-performing LightGBM model
- Interpret key fraud drivers using feature importance
- Showcase industry-ready ML workflow

---

## 🔧 Technologies Used

- Python (Pandas, NumPy, Scikit-learn models: LightGBM,CART-DecisionTreeClassifier,RandomForestClassifier)
- Seaborn & Matplotlib for visualizations
- Jupyter Notebooks for development

---

## 📦 Pipeline Overview

### ✅ 1. Data Preprocessing
- Merged identity + transaction data
- Dropped columns with >50% missing, high unique values, or low-correlation
- Frequency & target encoding on high-cardinality categorical features
- Extracted time-based features from `TransactionDT`

### ✅ 2. Feature Engineering
- Encoded `P_emaildomain`, 'ProductCD', `card` features
- Counted fraud frequency per category (smoothed)
- Filled missing values using robust defaults

### ✅ 3. Handling Imbalance (SMOTE)
- Applied **SMOTE** to oversample minority fraud class
- Ensures classifier sees enough fraud patterns during training

### ✅ 4. Model Training
- Trained LightGBM, DecisionTreeClassifier, RandomForestClassifier on resampled data
- Used AUC-ROC as evaluation metric
- Visualized top feature importances

---

## 📈 Evaluation

| Metric     | Value  |
|------------|--------|
| ROC AUC    | 0.91+  |
| Precision  | ~0.85  |
| Recall     | ~0.45  |
| F1-Score   | ~0.58  |

> Metrics reflect validation set, with resampling applied

---

## 🚀 Future Work

- Use LightGBM + SMOTE in a pipeline with cross-validation
- Deploy via Flask/Streamlit API
- Use `MLflow` for experiment tracking
- Run as an Airflow DAG for production deployment

---

## 📁 Project Structure

```
fraud-detection/
├── data/                  # (contains .gitignore so raw data isn't tracked)
│   └── .gitignore
├── notebooks/
│   └── HW2.ipynb
├── src/
│   └── feature_engineering.py
├── models/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧠 Author

Ha Do — Data Engineering & ML Engineering Candidate  
Let’s connect on [LinkedIn]([https://www.linkedin.com/in/ha-van-do/])

