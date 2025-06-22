from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix, recall_score, precision_score, f1_score, precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#load data 
df = pd.read_csv('/Users/eb2007/playground/bullpy/data_manipulation/Titanic-Dataset-cleaned.csv')

#handle missing values 
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#Encode categorical vars 
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

#scale numerical vars 
scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'Pclass']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

#prepare features for modelling 
features = ['Age', 'Fare', 'Pclass', 'Sex_encoded', 'Embarked_encoded']
x = df[features]
y_regression = df['Fare'] #regression prediction fare
y_classification = df['Survived'] #classification prediction survived 

#split data for classification 
x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(x, y_classification, test_size=0.2, random_state=42)
#split data for regression 
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x, y_regression, test_size=0.2, random_state=42)

#train classification model 
clf = RandomForestClassifier(random_state=42)
clf.fit(x_train_clf, y_train_clf)

#train regression model 
model = LinearRegression()
model.fit(x_train_reg, y_train_reg)

#predict 
y_pred_clf = clf.predict(x_test_clf)
y_pred_reg = model.predict(x_test_reg)

#evaluate 
print("===CLASSIFICATION MODEL EVAL LF===")
print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.3f}")
print(f"Precision: {precision_score(y_test_clf, y_pred_clf):.3f}")
print(f"Recall: {recall_score(y_test_clf, y_pred_clf):.3f}")
print(f"F1-Score: {f1_score(y_test_clf, y_pred_clf):.3f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test_clf, y_pred_clf))

#confusion matrix 
cm = confusion_matrix(y_test_clf, y_pred_clf)
print("\nConfusion Matrix:")
print(cm)

# Evaluate regression
print("\n=== REGRESSION MODEL EVALUATION ===")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.3f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.3f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.3f}")

#show predictions vs actual 
plt.figure(figsize=(12, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, label='Predicted')
plt.scatter(y_test_reg, y_test_reg, alpha=0.7, label='Actual')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Actual vs Predicted Fare')
plt.legend()
plt.show()

print(f"\nSample Predictions vs Actual:")
for i in range(min(5, len(y_test_reg))):
    print(f"Actual: {y_test_reg.iloc[i]:.2f}, Predicted: {y_pred_reg[i]:.2f}")