# 🚀 Intelligent CRM Lead Scoring System

This project is a Machine Learning-based Lead Scoring System that predicts the probability of a lead converting using CRM data.

It includes:
- Model training pipeline (`project.py`)
- Interactive web app using Streamlit (`project_a.py`)

---


## ⚙️ Features

- Data preprocessing (missing values, encoding, scaling)
- Multiple ML models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model evaluation using:
  - ROC-AUC
  - PR-AUC
- Feature importance visualization
- Interactive prediction UI (Streamlit)
- Model saving using joblib

---

## 📊 Dataset

The dataset (`Leadss_scoring.csv`) contains CRM lead information such as:
- Lead source
- Activity
- Time spent on website
- Page views
- Occupation

Target variable:
- `Converted` (1 = Converted, 0 = Not Converted)

---

## 🧠 Models Used

- Logistic Regression (baseline model)
- Random Forest (ensemble model)
- XGBoost (boosting model)

---

## ▶️ How to Run

### 1. Install Dependencies
pip install pandas numpy matplotlib scikit-learn xgboost streamlit joblib

---

### 2. Train Models
python project.py

This will:
- Train models
- Print evaluation metrics
- Save model files

---

### 3. Run Streamlit App
streamlit run project_a.py

---

## 🌐 Streamlit Features

- Dataset preview
- Missing value analysis
- Model comparison
- Feature importance charts
- Prediction system

---

## 🔮 Prediction

Steps:
1. Select model
2. Enter feature values
3. Click "Predict"
4. View conversion probability

---

## 💾 Saved Files

- lr_model.pkl
- rf_model.pkl
- xgb_model.pkl
- scaler.pkl
- encoders

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## 👩‍💻 Author

Tanvi Aneja  
B.Tech Robotics & AI Engineering