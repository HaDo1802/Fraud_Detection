
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

## 💼 Business Impact of the Model

The model addresses a critical business challenge: **fraudulent financial transactions** that result in millions of dollars in losses for online retailers, banks, and payment processors. By accurately identifying fraudulent transactions with high recall and AUC, the system:

- Helps reduce financial losses from unauthorized transactions.
- Enables real-time transaction flagging and intervention.
- Supports regulatory compliance and risk mitigation strategies.
- Improves customer trust by minimizing false positives (legitimate users being flagged).

### Example Use Case:
A flagged transaction with a high fraud probability could trigger a verification prompt or temporary hold, reducing the chances of loss while minimizing inconvenience for legitimate users.

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
## ⚠️ Model Limitations

Despite solid performance, the model has some limitations:

- **Imbalanced Data Risk**: Even with SMOTE, there's a chance of overfitting synthetic fraud patterns that may not generalize well.
- **Feature Obfuscation**: Many features (like V1–V339) are anonymized and may not reflect real-world interpretability.
- **Real-Time Limitations**: SMOTE and model retraining are batch processes; this may delay detection in real-time environments.
- **Dynamic Fraud Patterns**: Fraud tactics evolve. A model trained on static historical data might miss new fraud techniques.

---

## 🔧 Suggestions for Improvement

To make the model more robust and production-ready:

1. **Auto-Feature Engineering**: Use tools like `featuretools` or `Kats` to discover temporal or cross-entity relationships.
2. **Adversarial Validation**: Ensure train/test splits are from the same distribution to avoid data drift.
3. **Model Stacking**: Combine LightGBM with XGBoost, RandomForest or deep learning models using a meta-classifier.
4. **Online Learning**: Transition to an online learning setup (e.g., with River or Vowpal Wabbit) for real-time updates.
5. **Deploy Monitoring**: Set up logging for prediction drift, fraud false positives, and concept drift over time.

These strategies would better align the model with the dynamic nature of financial fraud and support both batch and real-time detection.

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
├── models/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧠 Author

Ha Do — Data Engineering & ML Engineering Candidate  
Let’s connect on [LinkedIn]: https://www.linkedin.com/in/ha-van-do/
---


