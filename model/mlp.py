import numpy as np

from model.layers import Dense, ReLU, Dropout
from model.loss import SoftmaxCrossEntropy

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