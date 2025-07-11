import pandas as pd

#creating dataframes
data = {
    'name': ['your nan', 'bull', 'terrance'],
    'age': [14,41, 14],
    'salary': [14000, 140, 140000]
}
df = pd.DataFrame(data)

# reading real data 
# df = pd.read_csv('data.csv')

#essential for understanding  data
print(df.shape) # dimensions  
print(df.head()) # first 5 rows 
print(df.info()) # statistical summary 
print(df.describe()) #data types and missing values

# accessing data
ages = df['age'] #single column 
subset = df[['name', 'age']] #multiple columns 
young_people = df[df['age'] < 14] #filtering

#data cleaning 
#handling missing values 
df_clean = df.dropna() 

#fill numeric columns with mean
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna('missing')

#data transformation 
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Old')
df['salary_normalized'] = (df['salary'] - df['salary'].mean()) / df['salary'].std()

