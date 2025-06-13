import numpy as np

#creating arrays (basis of all ML)
arr1d = np.array([1,2,3,4,5])
arr2d = np.array([[1,2,3], [4,5,6]])

#special arrays 
zeros = np.zeros((3,4))
ones = np.ones((2,3))
random_data = np.random.rand(100) #random data for test 

#array operations
#mathemaical operations (vectorized - very fast)

data = np.array([1, 2, 3, 4, 5])
squared = data ** 2
normalized = (data - np.mean(data)) / np.std(data)

#boolean indexing (crucial for filtering data)
large_values = data[data > 3]

#reshaping and indexing 
flat_data = np.array([1,2,3,4,5,6])
matrix = flat_data.reshape(2,3)

flat_data = np.array([1,2,3,4,5,6])
matrix = flat_data.reshape(2,3)

#indexing 
matrix[0,1] #first row second column
matrix[0,:] #all rows, first column

