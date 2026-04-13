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
from utils.persistence import load_all, save_all
from visualization.plots import plot_training_accuracy, plot_training_loss

TRAINING_PATH = "./Data/Training.csv"
TESTING_PATH = "./Data/Testing.csv"


def main():
    saved = load_all()

    df_train = remove_unused_column(load_csv(TRAINING_PATH))
    target, features = detect_columns(df_train)

    X_train = df_train[features].values.astype(np.float32)

    if saved is not None:
        print("Leaded saved model\n")

        model = saved["model"]
        encoder = saved["encoder"]

        conf_t = saved["thresholds"]["confidence"]
        ent_t = saved["thresholds"]["entropy"]

        verifier_thresholds = saved["verifier_thresholds"]

        y_train = encoder.transform(df_train[target].values)

        verifier = CaseBasedVerifier(X_train, y_train, k=5)
        verifier.set_thresholds(verifier_thresholds)
    
    else:
        print("No saved model found. Training...\n")

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(df_train[target].values)

        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        split = int(len(X_train) * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]
        
        Xtr, ytr = X_train[train_idx], y_train[train_idx]
        Xval, yval = X_train[val_idx], y_train[val_idx]

        model = MLP(X_train.shape[1], len(encoder.classes))
        opt = Adam()

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

        epochs = 50 
        for epoch in range(epochs):
            loss = model.train_batch(Xtr, ytr, opt)
            
            train_preds, _ = model.predict(Xtr)
            val_preds, _ = model.predict(Xval)
            
            val_logits = model.forward(Xval, training=False)
            v_loss = model.loss_fn.forward(val_logits, yval)
            
            t_acc = accuracy_score(ytr, train_preds)
            v_acc = accuracy_score(yval, val_preds)

            history['train_loss'].append(loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:02d} | Train Acc: {accuracy_score(ytr, train_preds):.4f} | Val Acc: {accuracy_score(yval, val_preds):.4f} | Loss: {loss:.4f}")

        plot_training_accuracy(history)
        plot_training_loss(history)

        _, val_probs = model.predict(Xval)
        conf_t = tune_confidence_threshold(yval, val_probs)
        ent_t = tune_entropy_threshold(yval, val_probs)

        print(f"\nConfidence threshold: {conf_t:.3f}")
        print(f"Entropy threshold: {ent_t:.3f}")

        df_test = remove_unused_column(load_csv(TESTING_PATH))

        X_test = df_test[features].values.astype(np.float32)
        y_test = encoder.transform(df_test[target].values)

        test_preds, _ = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)

        print(f"\nTest Accuracy: {test_acc:.4f}")

        verifier = CaseBasedVerifier(Xtr, ytr, k=5)

        thresholds = tune_verifier_thresholds(verifier, Xval, yval, model)
        verifier.set_thresholds(thresholds)

        print("\nVerifier thresholds learned:")
        print(f"Agreement Threshold  : {thresholds['agreement']:.2f}")
        print(f"Similarity Threshold : {thresholds['similarity']:.2f}")
        print(f"Confidence Threshold : {thresholds['confidence']:.2f}")

        save_all(model, encoder, {"confidence": conf_t, "entropy": ent_t}, thresholds)

        print("Model saved successfully.\n")

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
            print(f"Uncertain (entropy={ent:.2f}). Possible: {encoder.classes[t]} ({tc:.2f})")
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