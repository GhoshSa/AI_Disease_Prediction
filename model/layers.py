import numpy as np

from model.neuron import Neuron

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