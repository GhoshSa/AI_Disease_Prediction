import numpy as np

from utils.metrics import accuracy_score

def tune_confidence_threshold(y_true, probs):
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    thresholds = np.unique(max_probs)
    results = []

    for t in thresholds:
        accepted = max_probs >= t
        
        if np.sum(accepted) == 0:
            continue

        accuracy = accuracy_score(y_true[accepted], preds[accepted])
        coverage = np.mean(accepted)
        results.append((t, accuracy, coverage))
    
    accs = np.array([r[1] for r in results])
    covs = np.array([r[2] for r in results])

    score = accs - np.gradient(accs) * covs
    best_idx = np.argmax(score)

    return results[best_idx][0]

def tune_entropy_threshold(y_true, probs, percentile=90):
    preds = np.argmax(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    correct_entropy = entropy[preds == y_true]

    if len(correct_entropy) == 0:
        return np.max(entropy)

    return np.percentile(correct_entropy, percentile)