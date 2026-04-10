import numpy as np

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
        self.dw = np.dot(self.x.T, d_out) / len(d_out)
        self.db = np.mean(d_out)
        return np.outer(d_out, self.w)