import numpy as np

from utils.metrics import accuracy_score, entropy

def tune_confidence_threshold(y_true, probs):
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    thresholds = np.linspace(0.3, 0.95, 20)
    best_t = 0.5
    best_score = -1

    for t in thresholds:
        accepted = max_probs >= t

        if np.sum(accepted) < 10:
            continue

        acc = accuracy_score(y_true[accepted], preds[accepted])
        cov = np.mean(accepted)

        score = acc * cov

        if score > best_score:
            best_score = score
            best_t = t

    return best_t

def tune_entropy_threshold(y_true, probs, percentile=90):
    preds = np.argmax(probs, axis=1)
    ent = entropy(probs)
    correct_entropy = ent[preds == y_true]

    if len(correct_entropy) == 0:
        return np.max(ent)

    return np.percentile(correct_entropy, percentile)