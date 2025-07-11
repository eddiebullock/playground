import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function
        Formula: 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """Initialize weights and bias to zeros"""
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def compute_cost(self, X, y, weights, bias):
        """
        Compute the logistic regression cost function
        Formula: -1/m * sum(y*log(h) + (1-y)*log(1-h))
        where h is the predicted value (sigmoid output)
        """
        m = X.shape[0]  # number of training examples
        
        # Compute the predicted values
        z = np.dot(X, weights) + bias
        h = self.sigmoid(z)
        
        # Compute the cost
        cost = -1/m * np.sum(y * np.log(h + 1e-10) + (1-y) * np.log(1-h + 1e-10))
        return cost
    
    def compute_gradients(self, X, y, weights, bias):
        """Compute gradients for gradient descent"""
        m = X.shape[0]  # number of training examples
        
        # Compute the predicted values
        z = np.dot(X, weights) + bias
        h = self.sigmoid(z)
        
        # Compute gradients
        dw = 1/m * np.dot(X.T, (h - y))
        db = 1/m * np.sum(h - y)
        
        return dw, db
    
    def gradient_descent(self, X, y):
        """Perform gradient descent to optimize weights and bias"""
        m, n = X.shape  # m = number of examples, n = number of features
        
        # Initialize parameters
        self.initialize_parameters(n)
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Compute gradients
            dw, db = self.compute_gradients(X, y, self.weights, self.bias)
            
            # Update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
            # Compute cost and save for history
            if i % 100 == 0:
                cost = self.compute_cost(X, y, self.weights, self.bias)
                self.cost_history.append(cost)
                print(f"Cost after iteration {i}: {cost}")
    
    def fit(self, X, y):
        """Train the model with the given data"""
        self.gradient_descent(X, y)
        return self
    
    def predict_proba(self, X):
        """Return probability estimates for samples"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels for samples"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        """Return accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_decision_boundary(self, X, y):
        """Plot the decision boundary (works for 2D data only)"""
        if X.shape[1] != 2:
            print("Cannot plot decision boundary for data with more than 2 features")
            return
        
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Make predictions on the meshgrid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Logistic Regression Decision Boundary')
        plt.show()
    
    def plot_cost_history(self):
        """Plot the cost history"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, self.num_iterations, 100), self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.grid(True)
        plt.show()


# Example usage with Iris dataset
def main():
    # Load the Iris dataset and take only 2 classes for binary classification
    iris = load_iris()
    X = iris.data[:100, :2]  # For simplicity, only use the first 2 features
    y = iris.target[:100]    # Only use classes 0 and 1
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features (important for logistic regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegressionFromScratch(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot the decision boundary
    model.plot_decision_boundary(X_train, y_train)
    
    # Plot cost history
    model.plot_cost_history()
    
    # Compare with sklearn's logistic regression
    from sklearn.linear_model import LogisticRegression
    sklearn_model = LogisticRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_train_accuracy = sklearn_model.score(X_train, y_train)
    sklearn_test_accuracy = sklearn_model.score(X_test, y_test)
    print(f"Sklearn training accuracy: {sklearn_train_accuracy:.4f}")
    print(f"Sklearn test accuracy: {sklearn_test_accuracy:.4f}")


if __name__ == "__main__":
    main()