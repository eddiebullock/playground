from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('/Users/eb2007/playground/bullpy/data_manipulation/Titanic-Dataset-cleaned.csv')
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'Pclass']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

features = ['Age', 'Fare', 'Pclass', 'Sex_encoded', 'Embarked_encoded']
x = df[features]
y = df['Fare']  # or whatever target you want

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#cross validation 
model = RandomForestRegressor(random_state=42)
cv_scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

#hyperparameter tuning 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

