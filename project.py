#command+shift+p then select python interpretor 3.9.0

# ---------------------------
# Step 1: Load Libraries
# ---------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import joblib

# ---------------------------
# Step 1a: Load Dataset
# ---------------------------
df = pd.read_csv('Leadss_scoring.csv')  # Replace with your CSV path

print("First 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

# ---------------------------
# Step 2: Preprocessing
# ---------------------------

# Ensure target is integer
df['Converted'] = df['Converted'].astype(int)

# Drop unnecessary columns
df = df.drop(['ProspectID', 'LeadNumber'], axis=1)

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('Converted')
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Handle missing values
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna('Unknown')

# Encode categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Scale numeric features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ---------------------------
# Step 3: Train-Test Split
# ---------------------------
X = df.drop('Converted', axis=1)
y = df['Converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# ---------------------------
# Step 4a: Logistic Regression
# ---------------------------
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)

y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]

roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)
pr_auc_lr = auc(recall_lr, precision_lr)

print("\nLogistic Regression:")
print("ROC-AUC:", roc_auc_lr)
print("PR-AUC:", pr_auc_lr)

# ---------------------------
# Step 4b: Random Forest
# ---------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf)
pr_auc_rf = auc(recall_rf, precision_rf)

print("\nRandom Forest:")
print("ROC-AUC:", roc_auc_rf)
print("PR-AUC:", pr_auc_rf)

# ---------------------------
# Step 4c: XGBoost
# ---------------------------
xgb_model = xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_pred_proba_xgb)
pr_auc_xgb = auc(recall_xgb, precision_xgb)

print("\nXGBoost:")
print("ROC-AUC:", roc_auc_xgb)
print("PR-AUC:", pr_auc_xgb)

# ---------------------------
# Step 5: Feature Importance
# ---------------------------
# Random Forest
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(10,8))
rf_importances.plot(kind='barh')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.show()

# XGBoost
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(10,8))
xgb_importances.plot(kind='barh', color='orange')
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.show()

# ---------------------------
# Step 6: Save Models for Streamlit
# ---------------------------
joblib.dump(lr, 'lr_model.pkl')
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')

print("\nModels saved: lr_model.pkl, rf_model.pkl, xgb_model.pkl")

