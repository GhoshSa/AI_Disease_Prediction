import pickle
import os

MODEL_PATH = "saved/model.pkl"

def save_all(model, encoder, thresholds, verifier_thresholds):
    os.makedirs("saved", exist_ok=True)

    data = {
        "model": model,
        "encoder": encoder,
        "thresholds": thresholds,
        "verifier_thresholds": verifier_thresholds
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)


def load_all():
    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)