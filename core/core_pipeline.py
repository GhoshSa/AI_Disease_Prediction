import numpy as np

from utils.data_utils import detect_columns, load_csv, remove_unused_column
from utils.encoding import LabelEncoder
from model.mlp import MLP
from model.optimization import Adam
from utils.metrics import accuracy_score
from visualization.plots import plot_confidence_vs_entropy, plot_entropy_distribution, plot_entropy_vs_symptoms
from thresholds.threshold import tune_confidence_threshold, tune_entropy_threshold
from interaction.conversation import ConversationPredictor

TRAINING_PATH = "./Data/Training.csv"
TESTING_PATH = "./Data/Testing.csv"

def main():
    df_train = remove_unused_column(load_csv(TRAINING_PATH))
    target, features = detect_columns(df_train)

    X_train = df_train[features].values.astype(np.float32)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df_train[target].values)

    idx = np.random.permutation(len(X_train))
    split = int(0.8 * len(X_train))
    Xtr, Xval = X_train[idx[:split]], X_train[idx[split:]]
    ytr, yval = y_train[idx[:split]], y_train[idx[split:]]

    model = MLP(X_train.shape[1], len(encoder.classes))
    opt = Adam()

    print("\nStarting training...\n")
    for e in range(50):
        perm = np.random.permutation(len(Xtr))
        X_tr_epoch = Xtr[perm]
        y_tr_epoch = ytr[perm]

        loss = model.train_batch(X_tr_epoch, y_tr_epoch, opt)
        
        if e == 0 or (e + 1) % 5 == 0:
            tr_pred, _ = model.predict(Xtr)
            va_pred, _ = model.predict(Xval)
            print(f"Epoch {e+1:03d} | Train Acc: {accuracy_score(ytr, tr_pred):.4f} | Val Acc: {accuracy_score(yval, va_pred):.4f} | Loss: {loss:.4f}")

    _, val_probs = model.predict(Xval)
    conf_t = tune_confidence_threshold(yval, val_probs)
    ent_t = tune_entropy_threshold(yval, val_probs)

    print(f"\nConfidence threshold: {conf_t:.3f}")
    print(f"Entropy threshold: {ent_t:.3f}")

    plot_confidence_vs_entropy(val_probs)
    plot_entropy_distribution(yval, val_probs)

    df_test = load_csv(TESTING_PATH)

    X_test = df_test[features].values.astype(np.float32)
    y_test = encoder.transform(df_test[target].values)

    test_preds, _ = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\nTest accuracy: {test_acc:.4f}")

    predictor = ConversationPredictor(model, encoder, features, conf_t, ent_t)

    print("\nEnter symptoms (reset / empty to exit)\n")
    while True:
        text = input("Symptoms: ").strip()
        if not text:
            break
        if text.lower() == "reset":
            predictor.reset()
            print("Reset\n")
            continue

        predictor.update(text)
        decision, t, tc, s, sc, ent = predictor.predict()

        if decision == "wait":
            print(f"Uncertain (entropy={ent:.2f}). Possible: {encoder.classes[t]} ({tc:.2f})")
        elif decision == "abstain":
            print(f"Low confidence. Possible: {encoder.classes[t]} ({tc:.2f})")
        else:
            print(f"\nFinal Diagnosis: {encoder.classes[t]} "
                  f"(confidence={tc:.2f}, entropy={ent:.2f})\n")
            plot_entropy_vs_symptoms(predictor.symptom_hist, predictor.entropy_hist)
            break