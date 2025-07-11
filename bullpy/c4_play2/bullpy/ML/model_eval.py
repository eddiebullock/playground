from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from matplotlib import pyplot as plt

# Load and preprocess data
df = pd.read_csv('/Users/eb2007/playground/bullpy/data_manipulation/Titanic-Dataset-cleaned.csv')
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

#---feature engineering---
#creating new features
df['age_squared'] = df['Age'] ** 2
df['fare_per_age'] = df['Fare'] / (df['Age'] + 1e-6)  # Avoid division by zero

features = ['Age', 'Fare', 'Pclass', 'Sex_encoded', 'Embarked_encoded', 'age_squared', 'fare_per_age']
x = df[features]
y = df['Fare']  

#polynomial features 
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

scaler = StandardScaler()
x_poly_scaled = scaler.fit_transform(x_poly)

#split data
x_train, x_test, y_train, y_test = train_test_split(x_poly_scaled, y, test_size=0.2, random_state=42)

#cross validation 
model = RandomForestRegressor(random_state=42)
cv_scores = cross_val_score(model, x_poly_scaled, y, cv=5, scoring='neg_mean_squared_error')
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

#hyperparameter tuning 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

#feature importance 
best_model = grid_search.best_estimator_
importances = best_model.feature_importances_
for name, importance in zip(poly.get_feature_names_out(features), importances):
    print(f"{name}: {importance:.4f}")

#plot feature importance 
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), poly.get_feature_names_out(features))
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Random Forest Regression')
plt.show()
