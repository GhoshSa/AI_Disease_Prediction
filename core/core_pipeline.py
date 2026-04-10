import numpy as np

from utils.data_utils import detect_columns, load_csv, remove_unused_column
from utils.encoding import LabelEncoder
from model.mlp import MLP
from model.optimization import Adam
from utils.metrics import accuracy_score
from thresholds.threshold import tune_confidence_threshold, tune_entropy_threshold
from interaction.conversation import ConversationPredictor
from verification.verifier import CaseBasedVerifier
from thresholds.verifier_thresholds import tune_verifier_thresholds
from utils.metrics import entropy

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

            print(f"Epoch {e+1:03d} | "
                  f"Train Acc: {accuracy_score(ytr, tr_pred):.4f} | "
                  f"Val Acc: {accuracy_score(yval, va_pred):.4f} | "
                  f"Loss: {loss:.4f}")

    # Threshold tuning (conversation)
    _, val_probs = model.predict(Xval)
    conf_t = tune_confidence_threshold(yval, val_probs)
    ent_t = tune_entropy_threshold(yval, val_probs)

    print(f"\nConfidence threshold: {conf_t:.3f}")
    print(f"Entropy threshold: {ent_t:.3f}")

    # TEST PHASE
    df_test = remove_unused_column(load_csv(TESTING_PATH))

    X_test = df_test[features].values.astype(np.float32)
    y_test = encoder.transform(df_test[target].values)

    test_preds, _ = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\nTest Accuracy: {test_acc:.4f}")

    # VERIFIER
    verifier = CaseBasedVerifier(Xtr, ytr, k=5)

    thresholds = tune_verifier_thresholds(verifier, Xval, yval, model)
    verifier.set_thresholds(thresholds)

    print("\nVerifier thresholds learned:")
    print(f"Agreement Threshold  : {thresholds['agreement']:.2f}")
    print(f"Similarity Threshold : {thresholds['similarity']:.2f}")
    print(f"Confidence Threshold : {thresholds['confidence']:.2f}")

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

        if text.lower() == "force":
            _, probs = model.predict(predictor.vector)
            probs = probs[0]

            top2 = np.argsort(probs)[-2:][::-1]
            t, s = top2
            tc, sc = probs[t], probs[s]

            ent = entropy(probs.reshape(1, -1))[0]

            print("\nForced Prediction Triggered")

            print(f"\nModel Diagnosis: {encoder.classes[t]} (confidence={tc:.2f}, entropy={ent:.2f})")

            verdict, info = verifier.verify(t, predictor.vector[0], tc)

            if verdict == "accept":
                print(f"Verified Diagnosis: {encoder.classes[t]}\n")

            elif verdict == "uncertain":
                print(f"Verifier Uncertain: {info['reason']}")
                print(f"Possible: {encoder.classes[t]} ({tc:.2f}), {encoder.classes[s]} ({sc:.2f})\n")

            else:
                print(f"Verifier Rejected: {info['reason']}")
                print(f"Suggested: {encoder.classes[info['suggested']]}")
                print(f"Model said: {encoder.classes[t]} ({tc:.2f})\n")

            break

        predictor.update(text)

        decision, t, tc, s, sc, ent = predictor.predict()

        if decision == "wait":
            print(f"Uncertain (entropy={ent:.2f})")
            print("Not enough information yet...\n")
            continue

        elif decision == "abstain":
            print("Low confidence")
            print("Provide more symptoms...\n")
            continue

        elif decision == "predict":

            print(f"\nModel Diagnosis: {encoder.classes[t]} "
                  f"(confidence={tc:.2f}, entropy={ent:.2f})")

            verdict, info = verifier.verify(t, predictor.vector[0], tc)

            if verdict == "accept":
                print(f"Verified Diagnosis: {encoder.classes[t]}\n")

            elif verdict == "uncertain":
                print(f"Verifier Uncertain: {info['reason']}")
                print(f"Possible: {encoder.classes[t]} ({tc:.2f}), "
                      f"{encoder.classes[s]} ({sc:.2f})\n")

            else:
                print(f"Verifier Rejected: {info['reason']}")
                print(f"Suggested: {encoder.classes[info['suggested']]}")
                print(f"Model said: {encoder.classes[t]} ({tc:.2f})\n")

            break