import numpy as np

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