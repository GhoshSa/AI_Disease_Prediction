import numpy as np

class CaseBasedVerifier:
    def __init__(self, X_train, y_train, k=5):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

        self.agreement_threshold = 0.6
        self.similarity_threshold = 0.4
        self.conf_threshold = 0.5
        self.min_symptoms = 2

    def set_thresholds(self, thresholds):
        self.agreement_threshold = thresholds["agreement"]
        self.similarity_threshold = thresholds["similarity"]
        self.conf_threshold = thresholds["confidence"]

    def similarity(self, x1, x2):
        intersection = np.sum((x1 == 1) & (x2 == 1))
        union = np.sum((x1 == 1) | (x2 == 1))
        return intersection / union if union != 0 else 0.0

    def get_top_k_cases(self, input_vec):
        sims = []
        for i in range(len(self.X_train)):
            sim = self.similarity(self.X_train[i], input_vec)
            sims.append((sim, self.y_train[i]))

        sims.sort(reverse=True, key=lambda x: x[0])
        return sims[:self.k]

    def verify_raw(self, input_vec):
        top_cases = self.get_top_k_cases(input_vec)

        weighted_votes = {}
        for sim, label in top_cases:
            weighted_votes[label] = weighted_votes.get(label, 0) + sim

        best_label = max(weighted_votes, key=weighted_votes.get)
        total_weight = sum(weighted_votes.values())

        agreement = weighted_votes[best_label] / total_weight if total_weight > 0 else 0.0
        similarity = np.mean([sim for sim, _ in top_cases])

        return best_label, {
            "agreement": agreement,
            "similarity": similarity,
            "top_cases": top_cases
        }

    def verify(self, pred_class, input_vec, confidence=None):
        num_symptoms = int(np.sum(input_vec))

        if num_symptoms < self.min_symptoms:
            return "uncertain", {"reason": "Insufficient symptoms"}

        if confidence is not None and confidence < self.conf_threshold:
            return "uncertain", {"reason": "Low model confidence"}

        best_label, info = self.verify_raw(input_vec)

        agreement = info["agreement"]
        similarity = info["similarity"]

        if best_label != pred_class:
            return "reject", {"reason": "Similar cases disagree", **info}

        if agreement < self.agreement_threshold:
            return "uncertain", {"reason": "Low agreement", **info}

        if similarity < self.similarity_threshold:
            return "uncertain", {"reason": "Low similarity", **info}

        return "accept", {"reason": "Verified", **info}