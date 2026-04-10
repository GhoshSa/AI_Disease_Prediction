import numpy as np
import re

from utils.metrics import entropy

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

        ent = entropy(probs.reshape(1, -1))[0]

        self.entropy_hist.append(ent)
        self.symptom_hist.append(int(np.sum(self.vector)))

        top2 = np.argsort(probs)[-2:][::-1]
        top, second = top2
        top_conf = probs[top]

        if ent > self.entropy_threshold:
            return "wait", top, probs[top], second, probs[second], ent

        if top_conf < self.conf_threshold:
            return "abstain", top, probs[top], second, probs[second], ent

        return "predict", top, probs[top], None, None, ent