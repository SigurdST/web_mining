
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
import pandas as pd
import os

# XGBoost avec bons paramètres
def run_xgboost_only(X, y):
    model = XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        use_label_encoder=False,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    return model

# Random Forest
def run_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X, y)
    return model

# LightGBM
def run_lightgbm(X, y):
    model = LGBMClassifier(n_estimators=200, max_depth=12, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

# Évaluation + Sauvegarde CSV
def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    # F1 + Kappa
    f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"\nEvaluation for {model_name}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"Cohen’s Kappa: {kappa:.4f}")

    # Save CSV
    save_classification_report(report, model_name)

def save_classification_report(report_dict, model_name):
    df = pd.DataFrame(report_dict).T.reset_index()
    df.rename(columns={'index': 'Class'}, inplace=True)
    path = f"results/{model_name}.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(path, index=False)
