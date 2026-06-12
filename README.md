# 🔬 Semiconductor Manufacturing Yield Prediction

A supervised machine learning pipeline to predict **Pass/Fail yield outcomes** in semiconductor manufacturing using high-dimensional sensor data. Built as part of a data science capstone project.

---

## 📌 Problem Statement

Modern semiconductor manufacturing involves hundreds of sensors monitoring every step of the process. Not all signals are equally informative — yet a single failure at any stage can result in defective chips. This project builds a classifier that predicts whether a production unit will **Pass or Fail** based on sensor readings, enabling early detection and reduced production costs.

---

## 📊 Dataset

- **Source:** SECOM Semiconductor Manufacturing Dataset
- **Features:** 590+ sensor measurements per production instance
- **Target:** Binary Pass/Fail yield label (`-1` = Pass, `1` = Fail)
- **Challenge:** High-dimensional data, significant missing values, and severe class imbalance

---

## ⚙️ ML Pipeline

```
Raw Sensor Data
      │
      ▼
Missing Value Imputation (Median)
      │
      ▼
Outlier Treatment (IQR Capping)
      │
      ▼
Class Imbalance Handling (SMOTE)
      │
      ▼
Train/Test Split (75/25, Stratified)
      │
      ▼
Feature Scaling (StandardScaler)
      │
      ▼
Model Training & Hyperparameter Tuning (GridSearchCV)
      │
      ▼
Evaluation → Best Model Saved
```

---

## 🤖 Models Trained

| Model | Tuning | Notes |
|---|---|---|
| Logistic Regression | Default | Baseline classifier |
| Random Forest | GridSearchCV (`n_estimators`, `max_depth`) | Ensemble, handles non-linearity |
| **SVM (RBF Kernel)** | **GridSearchCV (`C`, `kernel`)** | **Best performing model ✅** |

All models evaluated on accuracy, precision, recall, and F1-score.

---

## 🏆 Results

**SVM achieved the highest accuracy** across all models tested, with strong and balanced precision-recall scores on both the Pass and Fail classes — demonstrating reliable performance on an inherently imbalanced dataset.

> The combination of SMOTE oversampling + StandardScaler + SVM with RBF kernel proved most effective for this high-dimensional sensor classification task.

---

## 🗂️ Repository Structure

```
semiconductor-yield-prediction-ml/
│
├── capstone2.ipynb          # Full ML pipeline notebook
├── final_yield_prediction_model.pkl   # Saved SVM model
├── scaler.pkl               # Saved StandardScaler
├── signal-data.csv          # Dataset (sensor readings)
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

- **Data:** `pandas`, `numpy`
- **Preprocessing:** `sklearn.preprocessing`, `sklearn.impute`, `imblearn`
- **Modeling:** `sklearn` — Logistic Regression, Random Forest, SVM
- **Tuning:** `GridSearchCV`, `StratifiedKFold`
- **Saving:** `joblib`

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/12Shreya8/semiconductor-yield-prediction-ml.git
cd semiconductor-yield-prediction-ml
```

2. Install dependencies
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib joblib
```

3. Open the notebook
```bash
jupyter notebook capstone2.ipynb
```

4. Run all cells — the pipeline handles everything from data loading to model saving.

---

## 🔮 Future Work

- Feature selection / dimensionality reduction (PCA, LASSO)
- Add XGBoost and ensemble stacking
- Streamlit dashboard for real-time yield prediction
- Deployment via FastAPI or Streamlit Cloud

---

## 👩‍💻 Author

**Shreya** — CS Engineering student with a focus on AI/ML  
[GitHub](https://github.com/12Shreya8) · [LinkedIn](https://linkedin.com/in/shreya-yergol)
