import numpy as np

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