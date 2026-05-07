# baselines/baseline_models.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1": f1_score(y_test, preds, average="macro", zero_division=0),
    }


def run_all_baselines(X_train, y_train, X_test, y_test):
    results = {}

    lr = LogisticRegression(max_iter=1000)
    results["Logistic Regression"] = evaluate_model(lr, X_train, y_train, X_test, y_test)

    svm = SVC()
    results["SVM"] = evaluate_model(svm, X_train, y_train, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    results["Random Forest"] = evaluate_model(rf, X_train, y_train, X_test, y_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    results["KNN"] = evaluate_model(knn, X_train, y_train, X_test, y_test)

    return results


def print_results(results):
    print("\nBaseline Model Results: \n")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1']:.4f}\n")