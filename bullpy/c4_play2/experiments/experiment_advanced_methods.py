import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

# Load data
# Use the preprocessed data file
# (Update the path as needed)
df = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_processed.csv')

# Drop columns that could leak data
drop_cols = [col for col in df.columns if col.startswith('diagnosis_') or col.startswith('autism_diagnosis_') or col in ['userid', 'repeat']]
X = df.drop(columns=drop_cols + ['autism_target'], errors='ignore')
y = df['autism_target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Threshold moving
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_test, probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
best_thresh = thresholds[np.argmax(f1s)]
print(f"Best threshold: {best_thresh}")
y_pred_thresh = (probs >= best_thresh).astype(int)
print(f"F1 with threshold moving: {f1_score(y_test, y_pred_thresh)}")

# Ensemble for imbalance (HPC)
# brf = BalancedRandomForestClassifier()
# brf.fit(X_train, y_train)
# print("Balanced RF F1:", f1_score(y_test, brf.predict(X_test)))

# Easy Ensemble (HPC)
# eec = EasyEnsembleClassifier()
# eec.fit(X_train, y_train)
# print("Easy Ensemble F1:", f1_score(y_test, eec.predict(X_test)))

# Anomaly detection framing (Isolation Forest)
iso = IsolationForest(contamination=float(sum(y_train)/len(y_train)))
iso.fit(X_train)
y_pred_anom = (iso.predict(X_test) == -1).astype(int)
print("Isolation Forest F1:", f1_score(y_test, y_pred_anom))

# Cost-sensitive learning
logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg.fit(X_train, y_train)
print("Cost-sensitive Logistic Regression F1:", f1_score(y_test, logreg.predict(X_test)))
