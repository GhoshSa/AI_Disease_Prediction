import numpy as np

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