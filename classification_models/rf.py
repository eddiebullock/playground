import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold 
from collections import Counter 

#helper functions
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    prob_sq = (counts / counts.sum()) ** 2
    return 1 - prob_sq.sum()

def split(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]