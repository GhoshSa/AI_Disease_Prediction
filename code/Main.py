import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Paths
TRAINING_PATH = "../Data/Training.csv"
TESTING_PATH = "../Data/Testing.csv"

# Data Utilities
def load_csv(path):
    return pd.read_csv(path)

def remove_unused_column(df, column="Unnamed: 133"):
    if column in df.columns:
        df = df.drop(column, axis=1)
    return df

def detect_columns(df):
    target_candidates = ["disease", "prognosis"]
    target = None
    for col in df.columns:
        if col.lower() in target_candidates:
            target = col
            break
    if target is None:
        raise ValueError("Target column not found")
    features = [c for c in df.columns if c != target]
    return target, features

# Label Encoder
class LabelEncoder:
    def fit(self, labels):
        self.classes = sorted(list(set(labels)))
        self.class_to_int = {c: i for i, c in enumerate(self.classes)}
        return self

    def transform(self, labels):
        return np.array([self.class_to_int[l] for l in labels], dtype=np.int64)

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

# Helpers
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Neural Network
class Neuron:
    def __init__(self, n_inputs, init="he"):
        if init == "he":
            self.w = np.random.randn(n_inputs) * np.sqrt(2.0 / n_inputs)
        else:
            self.w = np.random.randn(n_inputs) * np.sqrt(1.0 / n_inputs)
        self.b = 0.0

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, d_out):
        self.dw = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out)
        return np.outer(d_out, self.w)

class Dense:
    def __init__(self, n_inputs, n_outputs, init="he"):
        self.neurons = [Neuron(n_inputs, init) for _ in range(n_outputs)]

    def forward(self, x):
        self.x = x
        return np.stack([n.forward(x) for n in self.neurons], axis=1)

    def backward(self, d_out):
        dx = np.zeros_like(self.x)
        for i, n in enumerate(self.neurons):
            dx += n.backward(d_out[:, i])
        return dx

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask

class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, x, training=True):
        if not training:
            return x
        self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask

class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        loss = -np.log(np.clip(self.probs[np.arange(len(y_true)), y_true], 1e-9, 1))
        return np.mean(loss)

    def backward(self, y_true):
        d = self.probs.copy()
        d[np.arange(len(y_true)), y_true] -= 1
        return d / len(y_true)

# Optimizer
class Adam:
    def __init__(self, lr=0.001, eps=1e-7, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2

    def update_dense(self, layer):
        for n in layer.neurons:
            if not hasattr(n, "m_w"):
                n.m_w = np.zeros_like(n.w)
                n.v_w = np.zeros_like(n.w)
                n.m_b = 0.0
                n.v_b = 0.0

            n.m_w = self.beta1 * n.m_w + (1 - self.beta1) * n.dw
            n.v_w = self.beta2 * n.v_w + (1 - self.beta2) * (n.dw ** 2)
            n.w -= self.lr * n.m_w / (np.sqrt(n.v_w) + self.eps)

            n.m_b = self.beta1 * n.m_b + (1 - self.beta1) * n.db
            n.v_b = self.beta2 * n.v_b + (1 - self.beta2) * (n.db ** 2)
            n.b -= self.lr * n.m_b / (np.sqrt(n.v_b) + self.eps)

# MLP Model
class MLP:
    def __init__(self, input_dim, n_classes):
        self.layers = [
            Dense(input_dim, 64),
            ReLU(),
            Dropout(0.2),
            Dense(64, 32),
            ReLU(),
            Dropout(0.2),
            Dense(32, n_classes, "xavier")
        ]
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x, training=True):
        for l in self.layers:
            x = l.forward(x, training) if isinstance(l, Dropout) else l.forward(x)
        return x

    def train_batch(self, x, y, optimizer):
        logits = self.forward(x, True)
        loss = self.loss_fn.forward(logits, y)
        d = self.loss_fn.backward(y)
        for l in reversed(self.layers):
            if hasattr(l, "backward"):
                d = l.backward(d)
        for l in self.layers:
            if isinstance(l, Dense):
                optimizer.update_dense(l)
        return loss

    def predict(self, x):
        logits = self.forward(x, False)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return np.argmax(probs, axis=1), probs

# Threshold Tuning (Confidence + Entropy)
def tune_confidence_threshold(y_true, probs, min_coverage=0.4):
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    best_score = -1
    best_t = 0.5

    for t in np.linspace(0.1, 0.9, 30):
        accepted = max_probs >= t
        coverage = np.mean(accepted)
        if coverage < min_coverage:
            continue

        acc = np.mean(preds[accepted] == y_true[accepted])
        score = acc * coverage

        if score > best_score:
            best_score = score
            best_t = t

    return best_t

def tune_entropy_threshold(y_true, probs, percentile=90):
    preds = np.argmax(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    correct_entropy = entropy[preds == y_true]

    if len(correct_entropy) == 0:
        return np.max(entropy)

    return np.percentile(correct_entropy, percentile)

# Conversation Module
class ConversationPredictor:
    def __init__(self, model, encoder, symptom_columns, confidence_threshold, entropy_threshold):
        self.model = model
        self.encoder = encoder
        self.conf_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.symptom_index = {s.lower(): i for i, s in enumerate(symptom_columns)}
        self.reset()

    def reset(self):
        self.vector = np.zeros((1, len(self.symptom_index)), dtype=np.float32)
        self.entropy_hist = []
        self.symptom_hist = []

    def update(self, text):
        tokens = re.split(r"[,\.;]", text.lower())
        for t in tokens:
            t = t.strip()
            if t in self.symptom_index:
                self.vector[0, self.symptom_index[t]] = 1.0

    def predict(self):
        _, probs = self.model.predict(self.vector)
        probs = probs[0]

        entropy = -np.sum(probs * np.log(probs + 1e-9))

        self.entropy_hist.append(entropy)
        self.symptom_hist.append(int(np.sum(self.vector)))

        top2 = np.argsort(probs)[-2:][::-1]
        top, second = top2
        top_conf = probs[top]

        if entropy > self.entropy_threshold:
            return "wait", top, probs[top], second, probs[second], entropy

        if top_conf < self.conf_threshold:
            return "abstain", top, probs[top], second, probs[second], entropy

        return "predict", top, probs[top], None, None, entropy

# Visualization
def plot_entropy_vs_symptoms(symptoms, entropy):
    plt.figure()
    plt.plot(symptoms, entropy, marker="o")
    plt.xlabel("Number of Symptoms")
    plt.ylabel("Entropy")
    plt.title("Entropy Reduction with Evidence")
    plt.savefig("entropy_vs_symptoms.png", dpi=300)
    plt.close()

def plot_confidence_vs_entropy(probs):
    plt.figure()
    plt.scatter(np.max(probs, axis=1),
                -np.sum(probs * np.log(probs + 1e-9), axis=1),
                alpha=0.5)
    plt.xlabel("Max Confidence")
    plt.ylabel("Entropy")
    plt.title("Confidence vs Entropy")
    plt.savefig("confidence_vs_entropy.png", dpi=300)
    plt.close()

def plot_entropy_distribution(y, probs):
    pred = np.argmax(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    plt.hist(entropy[pred == y], bins=30, alpha=0.7, label="Correct")
    plt.hist(entropy[pred != y], bins=30, alpha=0.7, label="Incorrect")
    plt.legend()
    plt.xlabel("Entropy")
    plt.title("Entropy Distribution")
    plt.savefig("entropy_distribution.png", dpi=300)
    plt.close()

# Main Pipeline
def main():
    df = remove_unused_column(load_csv(TRAINING_PATH))
    target, features = detect_columns(df)

    X = df[features].values.astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target].values)

    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = MLP(X.shape[1], len(encoder.classes))
    optimizer = Adam()

    print("\nStarting training...\n")
    for epoch in range(50):
        loss = model.train_batch(X_tr, y_tr, optimizer)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            train_preds, _ = model.predict(X_tr)
            train_acc = accuracy_score(y_tr, train_preds)

            val_preds, _ = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            print(f"Epoch {epoch+1:03d} | Train accuracy: {train_acc:.4f} | Train Loss: {loss:.4f} | Validation accuracy: {val_acc:.4f}")

    _, val_probs = model.predict(X_val)

    conf_t = tune_confidence_threshold(y_val, val_probs)
    ent_t = tune_entropy_threshold(y_val, val_probs)

    print(f"\nLearned confidence threshold: {conf_t:.3f}")
    print(f"Learned entropy threshold: {ent_t:.3f}")

    plot_confidence_vs_entropy(val_probs)
    plot_entropy_distribution(y_val, val_probs)

    predictor = ConversationPredictor(
        model=model,
        encoder=encoder,
        symptom_columns=features,
        confidence_threshold=conf_t,
        entropy_threshold=ent_t
    )

    print("\nEnter symptoms separated by commas (reset / empty to exit)\n")
    while True:
        text = input("Symptoms: ").strip()
        if not text:
            break
        if text.lower() == "reset":
            predictor.reset()
            print("Reset\n")
            continue

        predictor.update(text)
        decision, t, tc, s, sc, entropy = predictor.predict()

        if decision == "wait":
            print(f"Uncertain (entropy={entropy:.2f}). Possible: {encoder.classes[t]} ({tc:.2f})")
        elif decision == "abstain":
            print(f"Low confidence. Possible: {encoder.classes[t]} ({tc:.2f})")
        else:
            print(f"\nFinal Diagnosis: {encoder.classes[t]} (confidence={tc:.2f}, entropy={entropy:.2f})\n")
            plot_entropy_vs_symptoms(predictor.symptom_hist, predictor.entropy_hist)
            break

# Entry
if __name__ == "__main__":
    main()