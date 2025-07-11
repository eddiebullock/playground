from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

#generate sample data 
np.random.seed(50)
x = np.random.rand(100, 1) * 10 #feature
y = 2 * x.flatten() + 1 + np.random.rand(100) #target with noise

#split data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#creat and train model 
model = LinearRegression()
model.fit(x_train, y_train)

#make predictions 
y_pred = model.predict(x_test)

#evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"mean squared error bro: {mse}")

#vis results
plt.scatter(x_test, y_test, alpha=0.7, label='actual')
plt.scatter(x_test, y_pred, alpha=0.7, label='predicted')
plt.legend()
plt.show()