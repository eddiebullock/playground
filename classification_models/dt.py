import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('/Users/eb2007/playground/data/raw/dt_scratch.csv')

# Separate features and labels
x = data[['Age', 'Income']].values
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
