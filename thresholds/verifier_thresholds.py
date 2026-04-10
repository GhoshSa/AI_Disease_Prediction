import numpy as np

def tune_verifier_thresholds(verifier, X_val, y_val, model):
    agreements = []
    similarities = []
    confidences = []
    correct_flags = []

    for i in range(len(X_val)):
        x = X_val[i:i+1]
        y_true = y_val[i]

        pred, probs = model.predict(x)
        pred = pred[0]
        probs = probs[0]

        confidence = probs[pred]

        best_label, info = verifier.verify_raw(x[0])

        agreements.append(info["agreement"])
        similarities.append(info["similarity"])
        confidences.append(confidence)
        correct_flags.append(int(pred == y_true))

    agreements = np.array(agreements)
    similarities = np.array(similarities)
    confidences = np.array(confidences)
    correct_flags = np.array(correct_flags)

    agreements_correct = agreements[correct_flags == 1]
    similarities_correct = similarities[correct_flags == 1]
    confidences_correct = confidences[correct_flags == 1]

    agreement_t = np.percentile(agreements_correct, 10)
    similarity_t = np.percentile(similarities_correct, 10)
    confidence_t = np.percentile(confidences_correct, 10)

    agreement_t *= 0.9
    similarity_t *= 0.85
    confidence_t *= 0.95

    agreement_t = np.clip(agreement_t, 0.5, 0.9)
    similarity_t = np.clip(similarity_t, 0.3, 0.8)
    confidence_t = np.clip(confidence_t, 0.4, 0.85)

    return {
        "agreement": float(agreement_t),
        "similarity": float(similarity_t),
        "confidence": float(confidence_t)
    }