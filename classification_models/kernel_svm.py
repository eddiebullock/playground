import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Create non-linear dataset
X, y = make_circles(
    n_samples=100,   # 100 total points
    factor=0.5,      # Inner circle has radius 0.5 of outer circle
    noise=0.2        # Add some randomness so it's not perfectly clean
)
y = np.where(y == 0, -1, 1)  # Convert labels to -1, 1

# Visualize raw data before training
"""plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("Raw Data: Two Concentric Circles")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()"""

# Kernel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3):
    return (1 + np.dot(x1, x2)) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

class KernelSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=100, kernel='rbf', **kernel_params):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.alphas = None
        self.b = 0
        self.X = None
        self.y = None

    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return linear_kernel(x1, x2)
        elif self.kernel == 'polynomial':
            return polynomial_kernel(x1, x2, **self.kernel_params)
        elif self.kernel == 'rbf':
            return rbf_kernel(x1, x2, **self.kernel_params)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.X = X
        self.y = y
        self.b = 0

        # Precompute the kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])

        # Gradient ascent on dual problem
        for _ in range(self.n_iters):
            for i in range(n_samples):
                margin = np.sum(self.alphas * self.y * K[:, i]) + self.b
                if y[i] * margin < 1:
                    self.alphas[i] += self.lr * (1 - y[i] * margin - self.lambda_param * self.alphas[i])
                else:
                    self.alphas[i] += self.lr * (-self.lambda_param * self.alphas[i])
            # Optional: update bias
            self.b = np.mean([y[i] - np.sum(self.alphas * self.y * K[:, i]) for i in range(n_samples)])

    def project(self, X_new):
        result = []
        for x in X_new:
            s = 0
            for alpha, x_i, y_i in zip(self.alphas, self.X, self.y):
                s += alpha * y_i * self._kernel_function(x_i, x)
            s += self.b
            result.append(s)
        return np.array(result)

    def predict(self, X_new):
        return np.sign(self.project(X_new))

# Train the model
svm = KernelSVM(kernel='rbf', gamma=0.1)  # Using RBF kernel with gamma=0.1
svm.fit(X, y)
preds = svm.predict(X)

# Accuracy
print(f"Accuracy: {np.mean(preds == y) * 100:.2f}%")

# Plotting function
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['#FFAAAA', '#AAAAFF'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.title(f"Kernel SVM ({model.kernel.upper()}) Decision Boundary")
    plt.show()

# Show the decision boundary
plot_decision_boundary(X, y, svm)
