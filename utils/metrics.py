import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-9), axis = -1)