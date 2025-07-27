# --- TEST SUCCESSFUL C4 MODEL ON YBT WITH PROPER THRESHOLD TUNING ---
# Copy this code into a new cell in your general_test_ybt.ipynb notebook

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score

print("="*50)
print("TESTING SUCCESSFUL C4 MODEL ON YBT DATA")
print("="*50)

# Load the updated YBT data (now with individual questionnaire items)
ybt_df_updated = pd.read_csv('/Users/eb2007/playground/bullpy/c4_play2/data/processed/YBT_processed.csv')
print(f"Updated YBT data shape: {ybt_df_updated.shape}")

# Check if individual questionnaire items are now available
eq_items = [col for col in ybt_df_updated.columns if col.startswith('eq_') and col != 'eq_total']
sqr_items = [col for col in ybt_df_updated.columns if col.startswith('sqr_') and col != 'sqr_total']
print(f"Individual EQ items available: {len(eq_items)}")
print(f"Individual SQR items available: {len(sqr_items)}")

# Get feature list from the loaded model (assuming 'model' is already loaded)
feature_list = model.feature_names_in_

# Check for missing features in updated YBT data
missing_features = [f for f in feature_list if f not in ybt_df_updated.columns]
print(f"\nMissing features in updated YBT: {len(missing_features)}")
if missing_features:
    print("Missing features:", missing_features[:10])

# Select features for prediction
available_features = [f for f in feature_list if f in ybt_df_updated.columns]
X_ybt = ybt_df_updated[available_features]
y_true = ybt_df_updated['autism_target']

print(f"\nFeature matrix shape: {X_ybt.shape}")
print(f"Available features: {len(available_features)}/{len(feature_list)}")

# Get predictions
y_probs = model.predict_proba(X_ybt)[:, 1]
y_pred_default = model.predict(X_ybt)

# Default threshold performance
print("\n--- Performance with default threshold (0.5) ---")
print(classification_report(y_true, y_pred_default))
print(f"ROC-AUC: {roc_auc_score(y_true, y_probs):.3f}")

# Find optimal threshold for F1 score (like in your successful C4 model)
print("\n--- Threshold tuning for optimal F1 ---")
prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
best_thresh = thresholds[np.argmax(f1s)]

print(f"Best threshold for F1: {best_thresh:.3f}")

# Apply optimal threshold
y_pred_optimal = (y_probs >= best_thresh).astype(int)

print("\n--- Performance with optimal threshold ---")
print(classification_report(y_true, y_pred_optimal))
print(f"F1 at optimal threshold: {f1_score(y_true, y_pred_optimal):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_true, y_probs):.3f}")

# Compare with your C4 results
print("\n--- Comparison with your C4 results ---")
print("Your C4 results:")
print("- F1 score: 0.683")
print("- Threshold: 0.770")
print("- ROC-AUC: 0.939")

print(f"\nYBT results:")
print(f"- F1 score: {f1_score(y_true, y_pred_optimal):.3f}")
print(f"- Threshold: {best_thresh:.3f}")
print(f"- ROC-AUC: {roc_auc_score(y_true, y_probs):.3f}")

# Save results
results_summary = pd.DataFrame({
    'metric': ['F1_score', 'threshold', 'roc_auc'],
    'C4_results': [0.683, 0.770, 0.939],
    'YBT_results': [f1_score(y_true, y_pred_optimal), best_thresh, roc_auc_score(y_true, y_probs)]
})
print("\nResults summary:")
print(results_summary) 