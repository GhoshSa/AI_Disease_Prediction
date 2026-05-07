import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils.metrics import precision_recall_f1, accuracy_score

def cross_validate_mlp(model_class, X, y, num_classes, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]

        model = model_class(X.shape[1], num_classes)

        from model.optimization import Adam
        opt = Adam()

        for _ in range(30):
            model.train_batch(Xtr, ytr, opt)

        preds, _ = model.predict(Xval)

        acc = accuracy_score(yval, preds)
        metrics = precision_recall_f1(yval, preds, num_classes)

        acc_list.append(acc)
        prec_list.append(metrics['precision'])
        rec_list.append(metrics['recall'])
        f1_list.append(metrics['f1'])

    return {
        "accuracy": (np.mean(acc_list), np.std(acc_list)),
        "precision": (np.mean(prec_list), np.std(prec_list)),
        "recall": (np.mean(rec_list), np.std(rec_list)),
        "f1": (np.mean(f1_list), np.std(f1_list)),
    }