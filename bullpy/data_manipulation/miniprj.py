import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())

#handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0])

print(df.info())
print(df.describe())
print(df.columns)
print(df.shape)
print(df.isnull().sum())

#create new column
df['fare_level'] = pd.cut(df['Fare'],
                          bins=[0, 10, 30, 100, 600],
                          labels=['cheapskate', 'medium', 'high', 'very_high'])

df.to_csv('Titanic-Dataset-cleaned.csv', index=False)