from data_loader import load_and_merge_all_data
from feature import engineer_features
from model import (
    run_xgboost_only, run_random_forest, run_lightgbm,
    evaluate_model
)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import pickle
import pandas as pd
import os

print("\nLoading data...")
df = load_and_merge_all_data()
print(f"{len(df)} tweets loaded")

# Filter Low / Medium / High
print("\nFiltering Low / Medium / High only...")
df = df[df['annotation_postPriority'].isin(['Low', 'Medium', 'High'])]
print(f"Distribution: {df['annotation_postPriority'].value_counts().to_dict()}")

# Feature Engineering
print("\nFeature engineering...")
df_features = engineer_features(df)
X = df_features.drop(columns=['annotation_postPriority'])
y = df_features['annotation_postPriority']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Create output directory
os.makedirs("results", exist_ok=True)

# === Train & Evaluate Models ===

# XGBoost
print("\nTraining XGBoost...")
xgb_model = run_xgboost_only(X_train_bal, y_train_bal)
evaluate_model(xgb_model, X_test, y_test, label_encoder, "results/xgboost")

# Random Forest
print("\nTraining Random Forest...")
rf_model = run_random_forest(X_train_bal, y_train_bal)
evaluate_model(rf_model, X_test, y_test, label_encoder, "results/random_forest")

# LightGBM
print("\nTraining LightGBM...")
lgb_model = run_lightgbm(X_train_bal, y_train_bal)
evaluate_model(lgb_model, X_test, y_test, label_encoder, "results/lightgbm")

# Save models
with open("model/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open("model/lgb_model.pkl", "wb") as f:
    pickle.dump(lgb_model, f)

# Save test data
pd.Series(y_test).to_csv("data/y_test.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)