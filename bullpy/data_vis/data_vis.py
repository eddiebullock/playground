import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Simpletons plot')
plt.show()

#histograms
data = np.random.normal(0, 1, 1000)
plt.hist(data, bins=30)
plt.title('distribution of simpletons data')
plt.show()

#scatter plots 
x = np.random.rand(100) * 0.1
y = 2 * x + np.random.rand(100) * 0.1

plt.scatter(x, y)
plt.xlabel('Feature')
plt.ylabel('target')
plt.title('Feature vs Target')
plt.show()

#multiple plots 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(x, bins=20)
ax1.set_title('Feature distribution')
ax2.scatter(x, y)
ax2.set_title('Relationship')
plt.tight_layout()
plt.show()
