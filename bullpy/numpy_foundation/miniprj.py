import numpy as np

arr2d = np.random.rand(5,4)

print("Original 2D array:")
print(arr2d)
print("\nShape of array:", arr2d.shape)

values_greater_than_0_5 = arr2d > 0.5

print("\nBoolean mask (True where values > 0.5)")
print(values_greater_than_0_5)

greater_values = arr2d[arr2d > 0.5]

print("\nValues greater than 0.5:")
print(greater_values)
