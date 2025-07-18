import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib
import os

# lead feature engineered data
input_path = '/home/eb2007/c4/data/processed/data_c4_balanced_fe.csv'
df = pd.read_csv(input_path)
X = df.drop(columns=['autism_target'])
y = df['autism_target']

# split for validation 
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ensure output directory exists
os.makedirs('/home/eb2007/c4/models/', exist_ok=True)
os.makedirs('/home/eb2007/c4/results/', exist_ok=True)

# random forest hyperparameter grid
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

# performance
rf_best = rf_grid.best_estimator_
y_val_pred = rf_best.predict(X_val)
print("Random Forest Validation Performance")
print(classification_report(y_val, y_val_pred))
print("accuracy:", accuracy_score(y_val, y_val_pred))
print("F1 score:", f1_score(y_val, y_val_pred))

# save the best model and results 
joblib.dump(rf_grid.best_estimator_, '/home/eb2007/c4/models/best_rf_model.joblib')
pd.DataFrame(rf_grid.cv_results_).to_csv('/home/eb2007/c4/results/rf_cv_results.csv', index=False)

# XGBoost hyperparameter grid
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'scale_pos_weight': [1, sum(y_train==0)/sum(y_train==1)]
}
xgb = XGBClassifier(random_state=42)
xgb_grid = GridSearchCV(xgb, xgb_param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)

# performance
xgb_best = xgb_grid.best_estimator_
y_val_pred = xgb_best.predict(X_val)
print("XGBoost Validation Performance")
print(classification_report(y_val, y_val_pred))
print("accuracy:", accuracy_score(y_val, y_val_pred))

print("F1 score:", f1_score(y_val, y_val_pred))
joblib.dump(xgb_grid.best_estimator_, '/home/eb2007/c4/models/best_xgb_model.joblib')
pd.DataFrame(xgb_grid.cv_results_).to_csv('/home/eb2007/c4/results/xgb_cv_results.csv', index=False)
