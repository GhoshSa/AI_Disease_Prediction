import numpy as np


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def entropy(probs, normalize=True):
    ent = -np.sum(probs * np.log(probs + 1e-9), axis=-1)

    if normalize:
        ent = ent / np.log(probs.shape[-1])

    return ent


def precision_recall_f1(y_true, y_pred, num_classes):
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "f1": float(np.mean(f1_list))
    }


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return cm