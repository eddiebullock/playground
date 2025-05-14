import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('/Users/eb2007/playground/data/raw/synthetic_dt_dataset.csv')

# Separate features and labels
x = data[['Age', 'Income', 'CreditScore', 'EducationLevel']].values
y = data["Buys"].values

def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    prob_sq = (counts / counts.sum()) ** 2
    return 1 - prob_sq.sum()

def split(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def information_gain(y, y_left, y_right):
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    return gini(y) - (weight_left * gini(y_left) + weight_right * gini(y_right))

def find_best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, _, y_left, y_right = split(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gain = information_gain(y, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

feature, threshold, gain = find_best_split(x, y)
print(f"Best feature: {feature}, Best threshold: {threshold}, Info Gain: {gain:.4f}")

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value #used for leaf nodes

    def is_leaf(self):
        return self.value is not None        

def build_tree(X, y, depth=0, max_depth=3, min_samples_split=2, min_impurity_decrease=1e-7):
    #if all label are the same, return a leaf node
    if len(np.unique(y)) == 1:
        return Node(value=y[0])
    
    # stop if depth limit is reached
    if depth >= max_depth or len(y) < min_samples_split:
        #predict the most common class
        majority_class = np.bincount(y).argmax()
        return Node(value=majority_class)
    
    #find the best split
    feature, threshold, gain = find_best_split(X, y)
    if gain <= min_impurity_decrease:
        majority_class = np.bincount(y).argmax()
        return Node(value=majority_class)
    
    #split the data
    X_left, X_right, y_left, y_right = split(X, y, feature, threshold)

    #recursively build the children
    left_node = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split, min_impurity_decrease)
    right_node = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split, min_impurity_decrease)

    return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)


def predict(tree, x):
    if tree.is_leaf():
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)
    
#build the tree
tree = build_tree(x, y)

# test prediction
for sample in x:
    print(f"Input: {sample}, Predicted: {predict(tree, sample)}")

def print_tree(node, depth=0):
    indent = " " * depth
    if node.is_leaf():
        print(f"{indent}Predict: {node.value}")
        return
    
    print(f"{indent}Feature: {node.feature} <= {node.threshold}")
    print_tree(node.left, depth + 1)
    print(f"{indent}Feature {node.feature} > {node.threshold}")
    print_tree(node.right, depth + 1)

print_tree(tree)

predictions = [predict(tree, sample) for sample in x]
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2f}")

#wrap into class
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(x, y)
predictions = tree.predict(x)

# fit scikit-learn model
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(x, y)

# compare predictions
sk_preds = clf.predict(x)
sk_acc = accuracy_score(y, sk_preds)

print("\n--- scikit-learn DecisionTreeClassifier ---")
for i, sample in enumerate(x):
    print(f"Input: {sample}, Predicted: {sk_preds[i]}")
print(f"Accuracy: {sk_acc:.2f}")

class MyDecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2, min_impurity_decrease=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def fit(self, X, y):
        self.tree = build_tree(
            X, y,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease
        )

    def predict(self, X):
        return [predict(self.tree, sample) for sample in X]

    def print_tree(self):
        print_tree(self.tree)

#use youre custom class
my_tree = MyDecisionTreeClassifier(max_depth=3)
my_tree.fit(x, y)
my_preds = my_tree.predict(x)
my_acc = accuracy_score(y, my_preds)

print("\n--- My DecisionTreeClassifier ---")
for i, sample in enumerate(x):
    print(f"Input: {sample}, Predicted: {my_preds[i]}")
print(f"Accuracy: {my_acc:.2f}")    

