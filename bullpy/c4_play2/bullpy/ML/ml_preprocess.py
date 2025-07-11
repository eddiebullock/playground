from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

#create sample data 
x = np.random.rand(100, 3) #100 samples 3 features 

#scaling features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#encoding categorical vars
le = LabelEncoder()
categories = ['cat', 'dog', 'yournan', 'yourda', 'bird']
encoded = le.fit_transform(categories)
print(encoded) # [0 1 0 2 1]
