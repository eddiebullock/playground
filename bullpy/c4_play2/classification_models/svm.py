from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np 

X, y = make_blobs(n_samples=100, centers=2, random_state=6)
y = np.where(y == 0, -1, 1) # Convert to -1, 1

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

class LinearSVM:
    def __init__(self, learning_rate=0.001, n_iters=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * 2 * self.lambda_param * self.w
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)
    
svm = LinearSVM()
svm.fit(X, y)
predictions = svm.predict(X)

accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100:.2f}%")

def plot_decision_boundary(X, y, model):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.predict(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

plt.figure()
plot_decision_boundary(X, y, svm)
plt.title("Linear SVM Decision Boundary")
plt.show()

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3):
    return (1 + np.dot(x1, x2)) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    