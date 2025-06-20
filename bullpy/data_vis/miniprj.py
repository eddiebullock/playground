import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/eb2007/playground/bullpy/data_manipulation/Titanic-Dataset-cleaned.csv')

#check for missing values
print(df.isnull().sum())

#check for duplicate values
print(df.duplicated().sum())

#visualize the data
#distribution of age 
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.hist(df['Age'].dropna(), bins=20, alpha=0.7)
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# distribution by gender 
plt.subplot(2, 2, 2)
plt.hist(df['Fare'], bins=20, alpha=0.7)
plt.title('Fare distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')

#survival by gender
plt.subplot(2, 2, 3)
survival_by_gender = df.groupby(['Sex', 'Survived']).size().unstack()
survival_by_gender.plot(kind='bar')
plt.title('survival by gender')
plt.xlabel('Gender')
plt.ylabel('count')

#age vs fare relationship
plt.subplot(2, 2, 4)
plt.scatter(df['Age'], df['Fare'], alpha=0.5)
plt.title("age vs fare")
plt.xlabel('Age')
plt.ylabel('Fare')

plt.tight_layout()
plt.show()

#survival rate by passenger class
plt.figure(figsize=(8, 6))
survival_by_class = df.groupby(['Pclass', 'survived']).size().unstack()
survival_by_class.plot(kind='bar')
plt.title('survival by passenger class')
plt.xlabel('passenger class')
plt.ylabel('count')
plt.legend(['died', 'survived'])
plt.show()
