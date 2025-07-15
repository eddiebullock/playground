import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Load data
df = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_processed.csv')
autistic_df = df[df['autism_target'] == 1]
non_autistic_df = df[df['autism_target'] == 0]

# Downsample non-autistic to match autistic count
non_autistic_down = resample(non_autistic_df, replace=False, n_samples=len(autistic_df), random_state=42)
df_balanced = pd.concat([autistic_df, non_autistic_down])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

def train_model(input_path, model_path):
    # Load processed data
    df = pd.read_csv(input_path)

    # Drop columns that could leak data
    drop_cols = [col for col in df.columns if col.startswith('diagnosis_') or col.startswith('autism_diagnosis_') or col in ['userid', 'repeat']]
    X = df.drop(columns=drop_cols + ['autism_target'], errors='ignore')
    y = df['autism_target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Feature Selection: Top 20 features by Random Forest importance ---
    from sklearn.ensemble import RandomForestClassifier
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_train, y_train)
    importances = pd.Series(rf_selector.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(20).index
    print("\nTop 20 features by importance:")
    print(top_features)
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # Logistic Regression (with increased max_iter)
    print("\n--- Logistic Regression ---")
    clf_logreg = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    clf_logreg.fit(X_train, y_train)
    y_pred_logreg = clf_logreg.predict(X_test)
    y_probs_logreg = clf_logreg.predict_proba(X_test)[:, 1]
    from sklearn.metrics import classification_report, roc_auc_score
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

    # XGBoost Model
    print("\n--- XGBoost ---")
    from xgboost import XGBClassifier
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
    clf_xgb.fit(X_train, y_train)
    y_pred_xgb = clf_xgb.predict(X_test)
    y_probs_xgb = clf_xgb.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_xgb))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs_xgb))
    joblib.dump(clf_xgb, model_path.replace('.joblib', '_xgb.joblib'))

if __name__ == "__main__":
    # Save the balanced dataset to a temporary file for modeling
    balanced_path = "/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_matched_balanced.csv"
    df_balanced.to_csv(balanced_path, index=False)
    model_path = "/Users/eb2007/playground/bullpy/c4_play2/models/logreg.joblib"
    train_model(balanced_path, model_path)