# logistic_regression_complete.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, num_iterations=1000, 
                 lambda_reg=0, batch_size=None, early_stopping=False, patience=5):
        # Basic parameters
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        
        # L2 Regularization parameter
        self.lambda_reg = lambda_reg
        
        # Mini-batch parameter
        self.batch_size = batch_size
        
        # Early stopping parameters
        self.early_stopping = early_stopping
        self.patience = patience
        
        # Model parameters
        self.weights = None
        self.bias = None
        
        # History tracking
        self.cost_history = []
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """Initialize weights and bias to zeros"""
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def compute_cost(self, X, y, weights, bias):
        """Compute the logistic regression cost function with optional L2 regularization"""
        m = X.shape[0]  # number of training examples
        
        # Compute predicted values
        z = np.dot(X, weights) + bias
        h = self.sigmoid(z)
        
        # Compute cross-entropy loss
        cost = -1/m * np.sum(y * np.log(h + 1e-10) + (1-y) * np.log(1-h + 1e-10))
        
        # Add L2 regularization if lambda > 0
        if self.lambda_reg > 0:
            reg_term = (self.lambda_reg / (2 * m)) * np.sum(weights**2)
            cost += reg_term
            
        return cost
    
    def compute_gradients(self, X, y, weights, bias):
        """Compute gradients for gradient descent with optional L2 regularization"""
        m = X.shape[0]  # number of training examples
        
        # Compute predicted values
        z = np.dot(X, weights) + bias
        h = self.sigmoid(z)
        
        # Compute basic gradients
        dw = 1/m * np.dot(X.T, (h - y))
        db = 1/m * np.sum(h - y)
        
        # Add L2 regularization if lambda > 0
        if self.lambda_reg > 0:
            dw += (self.lambda_reg / m) * weights
            
        return dw, db
    
    def gradient_descent(self, X, y, X_val=None, y_val=None):
        """Perform gradient descent with mini-batches and early stopping if enabled"""
        m, n = X.shape  # m = number of examples, n = number of features
        
        # Initialize parameters
        self.initialize_parameters(n)
        
        # For early stopping
        best_val_cost = float('inf')
        best_weights = self.weights.copy()
        best_bias = self.bias
        counter = 0
        
        # Determine batch size
        batch_size = m if self.batch_size is None else min(self.batch_size, m)
        num_complete_batches = m // batch_size
        
        # Gradient descent loop
        for i in range(self.num_iterations):
            # Shuffle data for mini-batch
            if self.batch_size is not None:
                permutation = np.random.permutation(m)
                X_shuffled = X[permutation]
                y_shuffled = y[permutation]
            else:
                X_shuffled, y_shuffled = X, y
            
            epoch_cost = 0
            
            # Process mini-batches
            for batch in range(num_complete_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, m)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradients and update parameters
                dw, db = self.compute_gradients(X_batch, y_batch, self.weights, self.bias)
                self.weights = self.weights - self.learning_rate * dw
                self.bias = self.bias - self.learning_rate * db
                
                # Compute batch cost
                batch_cost = self.compute_cost(X_batch, y_batch, self.weights, self.bias)
                epoch_cost += batch_cost
            
            # Process remaining samples if not divisible
            if m % batch_size != 0 and self.batch_size is not None:
                start_idx = num_complete_batches * batch_size
                X_batch = X_shuffled[start_idx:]
                y_batch = y_shuffled[start_idx:]
                
                dw, db = self.compute_gradients(X_batch, y_batch, self.weights, self.bias)
                self.weights = self.weights - self.learning_rate * dw
                self.bias = self.bias - self.learning_rate * db
            
            # Record cost every 100 iterations
            if i % 100 == 0:
                current_cost = self.compute_cost(X, y, self.weights, self.bias)
                self.cost_history.append(current_cost)
                print(f"Cost after iteration {i}: {current_cost:.6f}")
                
                # Early stopping check
                if self.early_stopping and X_val is not None and y_val is not None:
                    val_cost = self.compute_cost(X_val, y_val, self.weights, self.bias)
                    print(f"Validation cost: {val_cost:.6f}")
                    
                    if val_cost < best_val_cost:
                        best_val_cost = val_cost
                        best_weights = self.weights.copy()
                        best_bias = self.bias
                        counter = 0
                    else:
                        counter += 1
                    
                    if counter >= self.patience:
                        print(f"Early stopping at iteration {i}")
                        self.weights = best_weights
                        self.bias = best_bias
                        break
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model with the given data"""
        # Create validation set if needed for early stopping but not provided
        if self.early_stopping and (X_val is None or y_val is None):
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.gradient_descent(X, y, X_val, y_val)
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
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # Make predictions on meshgrid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
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


class MultiClassLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, 
                 lambda_reg=0, batch_size=None, early_stopping=False, patience=5):
        # Store parameters to pass to binary classifiers
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        
        # List to store binary classifiers
        self.models = []
        self.classes = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        # Store unique classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Train one binary classifier for each class (one-vs-rest)
        for i, current_class in enumerate(self.classes):
            print(f"Training classifier for class {current_class} ({i+1}/{n_classes})")
            
            # Create binary labels (1 for current class, 0 for others)
            binary_y = (y == current_class).astype(int)
            
            # Create validation binary labels if needed
            binary_y_val = None
            if X_val is not None and y_val is not None:
                binary_y_val = (y_val == current_class).astype(int)
            
            # Create and train binary classifier
            model = LogisticRegressionFromScratch(
                learning_rate=self.learning_rate,
                num_iterations=self.num_iterations,
                lambda_reg=self.lambda_reg,
                batch_size=self.batch_size,
                early_stopping=self.early_stopping,
                patience=self.patience
            )
            model.fit(X, binary_y, X_val, binary_y_val)
            
            # Add trained classifier to list
            self.models.append(model)
        
        return self
    
    def predict_proba(self, X):
        # Get probabilities from each binary classifier
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        proba = np.zeros((n_samples, n_classes))
        
        for i, model in enumerate(self.models):
            proba[:, i] = model.predict_proba(X)
        
        # Normalize probabilities to ensure they sum to 1
        proba_sum = np.sum(proba, axis=1, keepdims=True)
        normalized_proba = proba / proba_sum
        
        return normalized_proba
    
    def predict(self, X):
        # Get class with highest probability
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        # Calculate accuracy
        predictions = self.predict(X)
        return np.mean(predictions == y)


def hyperparameter_tuning(X, y, param_grid=None):
    """Perform hyperparameter tuning and return best parameters"""
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1, 0.5],
            'num_iterations': [500, 1000, 2000],
            'lambda_reg': [0, 0.01, 0.1, 1.0],
            'batch_size': [None, 16, 32],  # None = full batch
        }
    
    # Split data into train, validation, test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Grid search
    print("Starting hyperparameter tuning...")
    results = []
    best_val_acc = 0
    best_params = None
    best_model = None
    
    # Test each combination of hyperparameters
    for lr in param_grid['learning_rate']:
        for iters in param_grid['num_iterations']:
            for lamb in param_grid['lambda_reg']:
                for bs in param_grid['batch_size']:
                    print(f"\nTesting: lr={lr}, iters={iters}, lambda={lamb}, batch_size={bs}")
                    
                    # Create and train model
                    model = LogisticRegressionFromScratch(
                        learning_rate=lr,
                        num_iterations=iters,
                        lambda_reg=lamb,
                        batch_size=bs
                    )
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    train_acc = model.score(X_train, y_train)
                    val_acc = model.score(X_val, y_val)
                    
                    print(f"Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")
                    
                    # Store results
                    results.append({
                        'learning_rate': lr,
                        'num_iterations': iters,
                        'lambda_reg': lamb,
                        'batch_size': bs,
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc
                    })
                    
                    # Update best parameters if this model is better
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = {
                            'learning_rate': lr,
                            'num_iterations': iters,
                            'lambda_reg': lamb,
                            'batch_size': bs
                        }
                        best_model = model
    
    # Print best parameters
    print("\n--- Best Parameters ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Parameters: {best_params}")
    
    # Test on test set
    test_acc = best_model.score(X_test, y_test)
    print(f"Test accuracy with best model: {test_acc:.4f}")
    
    # Compare with sklearn
    from sklearn.linear_model import LogisticRegression
    sklearn_model = LogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train)
    sklearn_val_acc = sklearn_model.score(X_val, y_val)
    sklearn_test_acc = sklearn_model.score(X_test, y_test)
    print(f"Sklearn validation accuracy: {sklearn_val_acc:.4f}")
    print(f"Sklearn test accuracy: {sklearn_test_acc:.4f}")
    
    # Visualize results (for learning rate and regularization only)
    plt.figure(figsize=(15, 10))
    
    # Plot effect of learning rate
    lr_results = {}
    for lr in param_grid['learning_rate']:
        lr_results[lr] = [r['val_accuracy'] for r in results if r['learning_rate'] == lr 
                       and r['lambda_reg'] == best_params['lambda_reg']
                       and r['batch_size'] == best_params['batch_size']]
    
    plt.subplot(1, 2, 1)
    for lr, accs in lr_results.items():
        if accs:  # Check if list is not empty
            plt.plot(param_grid['num_iterations'][:len(accs)], accs, marker='o', label=f'LR={lr}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of Learning Rate')
    plt.legend()
    plt.grid(True)
    
    # Plot effect of regularization
    lambda_results = {}
    for lamb in param_grid['lambda_reg']:
        lambda_results[lamb] = [r['val_accuracy'] for r in results if r['lambda_reg'] == lamb 
                             and r['learning_rate'] == best_params['learning_rate']
                             and r['batch_size'] == best_params['batch_size']]
    
    plt.subplot(1, 2, 2)
    for lamb, accs in lambda_results.items():
        if accs:  # Check if list is not empty
            plt.plot(param_grid['num_iterations'][:len(accs)], accs, marker='o', label=f'Lambda={lamb}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of Regularization')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return best_params, best_model


def main():
    """Main function to demonstrate functionality"""
    # Load dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Choose what to demonstrate
    demo_hyperparameter_tuning = True  # Set to True to run hyperparameter tuning
    demo_binary = True  # Set to True to demo binary classification
    demo_multiclass = True  # Set to True to demo multiclass classification
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Hyperparameter tuning
    if demo_hyperparameter_tuning:
        print("\n--- Hyperparameter Tuning ---")
        # Use only first 2 classes for binary classification in tuning
        binary_X = X[y < 2]
        binary_y = y[y < 2]
        best_params, _ = hyperparameter_tuning(binary_X, binary_y)
        learning_rate = best_params['learning_rate']
        num_iterations = best_params['num_iterations']
        lambda_reg = best_params['lambda_reg']
        batch_size = best_params['batch_size']
    else:
        # Default parameters if not tuning
        learning_rate = 0.1
        num_iterations = 1000
        lambda_reg = 0.1
        batch_size = None
    
    # Binary classification demo
    if demo_binary:
        print("\n--- Binary Classification Demo ---")
        # Use only first 2 features for binary classification
        binary_X_train = X_train[:, :2]
        binary_X_test = X_test[:, :2]
        binary_y_train = (y_train == 0).astype(int)  # Class 0 vs rest
        binary_y_test = (y_test == 0).astype(int)
        
        # Create and train model
        binary_model = LogisticRegressionFromScratch(
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            lambda_reg=lambda_reg,
            batch_size=batch_size,
            early_stopping=True,
            patience=3
        )
        binary_model.fit(binary_X_train, binary_y_train)
        
        # Evaluate model
        train_acc = binary_model.score(binary_X_train, binary_y_train)
        test_acc = binary_model.score(binary_X_test, binary_y_test)
        print(f"Binary classification - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
        
        # Plot decision boundary
        binary_model.plot_decision_boundary(binary_X_train, binary_y_train)
        
        # Plot cost history
        binary_model.plot_cost_history()
    
    # Multiclass classification demo
    if demo_multiclass:
        print("\n--- Multiclass Classification Demo ---")
        # Create and train model
        multi_model = MultiClassLogisticRegression(
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            lambda_reg=lambda_reg,
            batch_size=batch_size
        )
        multi_model.fit(X_train, y_train)
        
        # Evaluate model
        train_acc = multi_model.score(X_train, y_train)
        test_acc = multi_model.score(X_test, y_test)
        print(f"Multiclass classification - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
        
        # Compare with sklearn
        from sklearn.linear_model import LogisticRegression
        sklearn_model = LogisticRegression(multi_class='ovr')
        sklearn_model.fit(X_train, y_train)
        sklearn_train_acc = sklearn_model.score(X_train, y_train)
        sklearn_test_acc = sklearn_model.score(X_test, y_test)
        print(f"Sklearn - Train accuracy: {sklearn_train_acc:.4f}, Test accuracy: {sklearn_test_acc:.4f}")


if __name__ == "__main__":
    main()