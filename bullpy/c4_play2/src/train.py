import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

def train_model(input_path, model_path):
    # Load processed data
    df = pd.read_csv(input_path)

    # Define features and target
    diagnosis_cols = [col for col in df.columns if col.startswith('diagnosis_') and not 'autism' in col]
    autism_diag_cols = [col for col in df.columns if 'autism_diagnosis' in col]
    X = df.drop(columns=['autism_target'] + diagnosis_cols + autism_diag_cols)
    y = df['autism_target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Logistic Regression (with increased max_iter)
    print("\n--- Logistic Regression ---")
    clf_logreg = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    clf_logreg.fit(X_train, y_train)
    y_pred_logreg = clf_logreg.predict(X_test)
    y_probs_logreg = clf_logreg.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_logreg))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs_logreg))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf_logreg, model_path.replace('.joblib', '_logreg.joblib'))

    # Random Forest
    print("\n--- Random Forest ---")
    clf_rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    y_probs_rf = clf_rf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_rf))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs_rf))
    joblib.dump(clf_rf, model_path.replace('.joblib', '_rf.joblib'))

    # Logistic Regression + SMOTE
    print("\n--- Logistic Regression + SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    clf_logreg_smote = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    clf_logreg_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = clf_logreg_smote.predict(X_test)
    y_probs_smote = clf_logreg_smote.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_smote))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs_smote))

    # Logistic Regression + RandomOverSampler
    print("\n--- Logistic Regression + RandomOverSampler ---")
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    clf_logreg_ros = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    clf_logreg_ros.fit(X_train_ros, y_train_ros)
    y_pred_ros = clf_logreg_ros.predict(X_test)
    y_probs_ros = clf_logreg_ros.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_ros))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs_ros))

    # Logistic Regression + RandomUnderSampler
    print("\n--- Logistic Regression + RandomUnderSampler ---")
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    clf_logreg_rus = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    clf_logreg_rus.fit(X_train_rus, y_train_rus)
    y_pred_rus = clf_logreg_rus.predict(X_test)
    y_probs_rus = clf_logreg_rus.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_rus))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs_rus))

    # XGBoost Model
    print("\n--- XGBoost ---")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
    clf_xgb.fit(X_train, y_train)
    y_pred_xgb = clf_xgb.predict(X_test)
    y_probs_xgb = clf_xgb.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_xgb))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs_xgb))
    joblib.dump(clf_xgb, model_path.replace('.joblib', '_xgb.joblib'))

if __name__ == "__main__":
    input_path = "/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_processed.csv"
    model_path = "/Users/eb2007/playground/bullpy/c4_play2/models/logreg.joblib"
    train_model(input_path, model_path)