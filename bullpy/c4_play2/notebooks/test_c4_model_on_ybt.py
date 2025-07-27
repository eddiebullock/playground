import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score

def test_c4_model_on_ybt():
    """
    Test the successful C4 model on YBT data with proper threshold tuning
    """
    print("="*50)
    print("TESTING C4 MODEL ON YBT DATA")
    print("="*50)
    
    # Load data
    c4_path = "/Users/eb2007/playground/bullpy/c4_play2/data/processed/data_c4_processed.csv"
    ybt_path = "/Users/eb2007/playground/bullpy/c4_play2/data/processed/YBT_processed.csv"
    
    df_c4 = pd.read_csv(c4_path)
    df_ybt = pd.read_csv(ybt_path)
    
    print(f"C4 data shape: {df_c4.shape}")
    print(f"YBT data shape: {df_ybt.shape}")
    
    # Load the trained model (you'll need to save this from your feature engineering notebook)
    # For now, let's train a model on C4 data to simulate your successful model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Prepare C4 data for training
    exclude_cols = ['autism_target', 'userid']
    feature_cols = [col for col in df_c4.columns if col not in exclude_cols]
    
    X_c4 = df_c4[feature_cols]
    y_c4 = df_c4['autism_target']
    
    # Split C4 data
    X_train, X_test, y_train, y_test = train_test_split(
        X_c4, y_c4, test_size=0.2, stratify=y_c4, random_state=42
    )
    
    # Train model on C4 (simulating your successful model)
    print("\nTraining model on C4 data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Get feature names
    feature_names = model.feature_names_in_
    print(f"Model trained with {len(feature_names)} features")
    
    # Check feature alignment with YBT
    missing_features = [f for f in feature_names if f not in df_ybt.columns]
    print(f"\nMissing features in YBT: {len(missing_features)}")
    if missing_features:
        print("Missing features:", missing_features[:10])  # Show first 10
    
    # Find common features
    common_features = [f for f in feature_names if f in df_ybt.columns]
    print(f"Common features: {len(common_features)}")
    
    # Prepare YBT data with common features
    X_ybt = df_ybt[common_features]
    y_ybt = df_ybt['autism_target']
    
    print(f"\nYBT feature matrix shape: {X_ybt.shape}")
    
    # Test model on YBT data
    print("\n" + "="*50)
    print("MODEL PERFORMANCE ON YBT DATA")
    print("="*50)
    
    # Get predictions
    y_probs = model.predict_proba(X_ybt)[:, 1]
    y_pred_default = model.predict(X_ybt)
    
    # Default threshold performance
    print("Performance with default threshold (0.5):")
    print(classification_report(y_ybt, y_pred_default))
    print(f"ROC-AUC: {roc_auc_score(y_ybt, y_probs):.3f}")
    
    # Find optimal threshold for F1 score
    print("\n" + "="*50)
    print("THRESHOLD TUNING FOR OPTIMAL F1")
    print("="*50)
    
    prec, rec, thresholds = precision_recall_curve(y_ybt, y_probs)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_thresh = thresholds[np.argmax(f1s)]
    
    print(f"Best threshold for F1: {best_thresh:.3f}")
    
    # Apply optimal threshold
    y_pred_optimal = (y_probs >= best_thresh).astype(int)
    
    print("\nPerformance with optimal threshold:")
    print(classification_report(y_ybt, y_pred_optimal))
    print(f"F1 at optimal threshold: {f1_score(y_ybt, y_pred_optimal):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_ybt, y_probs):.3f}")
    
    # Compare with your C4 results
    print("\n" + "="*50)
    print("COMPARISON WITH C4 RESULTS")
    print("="*50)
    print("Your C4 results:")
    print("- F1 score: 0.683")
    print("- Threshold: 0.770")
    print("- ROC-AUC: 0.939")
    
    print(f"\nYBT results:")
    print(f"- F1 score: {f1_score(y_ybt, y_pred_optimal):.3f}")
    print(f"- Threshold: {best_thresh:.3f}")
    print(f"- ROC-AUC: {roc_auc_score(y_ybt, y_probs):.3f}")
    
    # Feature importance analysis
    print("\n" + "="*50)
    print("TOP FEATURES BY IMPORTANCE")
    print("="*50)
    
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(10)
    print("Top 10 features:")
    for feat, imp in top_features.items():
        print(f"  {feat}: {imp:.4f}")
    
    # Check if top features are available in YBT
    print(f"\nTop features available in YBT: {sum([f in df_ybt.columns for f in top_features.index])}/{len(top_features)}")
    
    return {
        'model': model,
        'best_threshold': best_thresh,
        'f1_score': f1_score(y_ybt, y_pred_optimal),
        'roc_auc': roc_auc_score(y_ybt, y_probs),
        'common_features': common_features,
        'missing_features': missing_features
    }

if __name__ == "__main__":
    results = test_c4_model_on_ybt() 