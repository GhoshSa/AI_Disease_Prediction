import numpy as np

from baselines.baseline_models import print_results, run_all_baselines
from utils.data_utils import add_noise, detect_columns, load_csv, remove_unused_column
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
from visualization.plots import plot_clean_vs_noisy, plot_training_accuracy, plot_training_loss
from utils.metrics import precision_recall_f1, confusion_matrix
from visualization.plots import plot_confusion_matrix
from evaluation.cross_validation import cross_validate_mlp

TRAINING_PATH = "./Data/Training.csv"
TESTING_PATH = "./Data/Testing.csv"

def main():
    saved = load_all()

    df_train = remove_unused_column(load_csv(TRAINING_PATH))
    target, features = detect_columns(df_train)

    X_train = df_train[features].values.astype(np.float32)

    if saved is not None:
        print("Loaded saved model\n")

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

        print("\nRunning 5-Fold Cross Validation...\n")

        cv_results = cross_validate_mlp(
            MLP,
            X_train,
            y_train,
            num_classes=len(encoder.classes),
            folds=5
        )

        print("Cross Validation Results:")
        for k, (mean, std) in cv_results.items():
            print(f"{k}: {mean:.4f} ± {std:.4f}")

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

        Xtest = X_test
        Ytest = y_test

        clean_preds, _ = model.predict(X_test)
        clean_acc = accuracy_score(y_test, clean_preds)

        num_classes = len(encoder.classes)

        clean_metrics = precision_recall_f1(y_test, clean_preds, num_classes)
        clean_cm = confusion_matrix(y_test, clean_preds, num_classes)

        print("\nClean Dataset...")
        print(f"Accuracy  : {clean_acc:.4f}")
        print(f"Precision : {clean_metrics['precision']:.4f}")
        print(f"Recall    : {clean_metrics['recall']:.4f}")
        print(f"F1 Score  : {clean_metrics['f1']:.4f}")

        plot_confusion_matrix(clean_cm, filename="confusion_matrix_clean.png", title="Confusion Matrix (Clean Data)", class_names=encoder.classes)

        X_noisy = add_noise(X_test, noise_level=0.1)

        noisy_preds, _ = model.predict(X_noisy)
        noisy_acc = accuracy_score(y_test, noisy_preds)

        noisy_metrics = precision_recall_f1(y_test, noisy_preds, num_classes)
        noisy_cm = confusion_matrix(y_test, noisy_preds, num_classes)

        print("\nAdded noise...")
        print(f"Accuracy  : {noisy_acc:.4f}")
        print(f"Precision : {noisy_metrics['precision']:.4f}")
        print(f"Recall    : {noisy_metrics['recall']:.4f}")
        print(f"F1 Score  : {noisy_metrics['f1']:.4f}")

        plot_confusion_matrix(noisy_cm, filename="confusion_matrix_noisy.png", title="Confusion Matrix (Noisy Data - 10%)", class_names=encoder.classes)
        
        test_preds, _ = model.predict(Xtest)
        test_acc = accuracy_score(Ytest, test_preds)

        print(f"\nTest Accuracy: {test_acc:.4f}")

        Xtest_noisy = add_noise(Xtest, noise_level=0.1)

        baseline_results = run_all_baselines(Xtr, ytr, Xtest_noisy, Ytest)
        print_results(baseline_results)

        verifier = CaseBasedVerifier(Xtr, ytr, k=5)

        thresholds = tune_verifier_thresholds(verifier, Xval, yval, model)
        verifier.set_thresholds(thresholds)

        print("\nVerifier thresholds learned:")
        print(f"Agreement Threshold  : {thresholds['agreement']:.2f}")
        print(f"Similarity Threshold : {thresholds['similarity']:.2f}")
        print(f"Confidence Threshold : {thresholds['confidence']:.2f}")

        clean_metrics['accuracy'] = clean_acc
        noisy_metrics['accuracy'] = noisy_acc

        plot_clean_vs_noisy(clean_metrics, noisy_metrics)

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