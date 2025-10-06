
#command to run: streamlit run project_a.py
# put yes or no for prediction 

# ---------------------------
# Step 1: Load Libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import joblib
import streamlit as st

# ---------------------------
# Step 1a: Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('Leadss_scoring.csv')  # Replace with your CSV path
    return df

df = load_data()

st.title("🚀 Intelligent CRM Lead Scoring System")
st.write("Predict the probability of a lead converting using CRM data.")

# Display Dataset and Missing Values
# Display Dataset Preview and Missing Values vertically
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Missing Values per Column")
st.dataframe(df.isnull().sum())

# ---------------------------
# Step 2: Preprocessing
# ---------------------------
df['Converted'] = df['Converted'].astype(int)
df = df.drop(['ProspectID', 'LeadNumber'], axis=1)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('Converted')
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna('Unknown')

# Encode categorical columns and save encoders
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f"{col}_encoder.pkl")

# Scale numeric features and save scaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
joblib.dump(scaler, 'scaler.pkl')

# ---------------------------
# Step 3: Train-Test Split
# ---------------------------
X = df.drop('Converted', axis=1)
y = df['Converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Step 4: Model Training
# ---------------------------
# Logistic Regression
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
joblib.dump(lr, 'lr_model.pkl')

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, 'rf_model.pkl')

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgb_model.pkl')

# ---------------------------
# Step 5: Model Evaluation
# ---------------------------
def evaluate_model(name, y_true, y_proba):
    roc = roc_auc_score(y_true, y_proba) * 100
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision) * 100
    st.write(f"**{name}** - ROC-AUC: {roc:.1f}%, PR-AUC: {pr_auc:.1f}%")

y_pred_lr = lr.predict_proba(X_test)[:,1]
y_pred_rf = rf.predict_proba(X_test)[:,1]
y_pred_xgb = xgb_model.predict_proba(X_test)[:,1]

st.subheader("📊 Model Evaluation Metrics")
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)

# ---------------------------
# Step 6: Feature Importance
# ---------------------------
st.subheader("🌟 Feature Importance")

# Random Forest & XGBoost
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True)

st.write("Random Forest Feature Importance")
st.bar_chart(rf_importances)

st.write("XGBoost Feature Importance")
st.bar_chart(xgb_importances)

# Logistic Regression
lr_importances = pd.Series(abs(lr.coef_[0]), index=X.columns).sort_values(ascending=True)
st.write("Logistic Regression Feature Importance")
st.bar_chart(lr_importances)
# ---------------------------
# Step 7: Streamlit Prediction Interface (Important Features Only)
# ---------------------------
st.subheader("🔮 Predict Conversion for a New Lead (Important Features Only)")

# Model selection
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])
selected_model = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb_model}[model_choice]

# Pick top features based on selected model
if model_choice == "Random Forest":
    top_features = rf_importances.sort_values(ascending=False).tail(5).index.tolist()
elif model_choice == "XGBoost":
    top_features = xgb_importances.sort_values(ascending=False).tail(5).index.tolist()
else:  # Logistic Regression
    top_features = lr_importances.sort_values(ascending=False).tail(5).index.tolist()

new_lead = {}

# Ask input only for top features
for col in top_features:
    if col in num_cols:
        new_lead[col] = st.number_input(f"{col}", value=0.0)
    else:  # categorical
        val = st.text_input(f"{col}", value="Unknown")
        if val in encoders[col].classes_:
            new_lead[col] = encoders[col].transform([val])[0]
        else:
            new_lead[col] = 0  # Unknown category

# Fill all remaining columns with defaults (0 for numeric, 0 for categorical)
for col in X.columns:
    if col not in new_lead:
        new_lead[col] = 0

# Convert to DataFrame in the same column order
input_df = pd.DataFrame([new_lead], columns=X.columns)

# Scale numeric columns
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict probability
if st.button("Predict Conversion Probability"):
    prob = selected_model.predict_proba(input_df)[:,1][0]
    st.metric(label="Predicted Conversion Probability", value=f"{prob*100:.2f}%")
