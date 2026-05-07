import numpy as np


class CaseBasedVerifier:
    def __init__(self, X_train, y_train, k=5):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        
        self.thresholds = {
            "agreement": 0.7,
            "similarity": 0.5,
            "confidence": 0.6
        }

    def set_thresholds(self, thresholds):
        self.thresholds.update(thresholds)
    
    def _compute_similarity(self, x_input, x_train):
        input_sum = np.sum(x_input)

        if input_sum == 0:
            return 0.0

        common = np.sum((x_input == 1) & (x_train == 1))
        return common / input_sum
    
    def verify_raw(self, x_input):
        similarities = []

        for i in range(len(self.X_train)):
            sim = self._compute_similarity(x_input, self.X_train[i])
            similarities.append((sim, self.y_train[i]))

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[:self.k]

        sims = [s for s, _ in top_k]
        labels = [l for _, l in top_k]

        avg_similarity = np.mean(sims)

        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1

        majority_class = max(counts, key=counts.get)
        agreement = counts[majority_class] / self.k

        return majority_class, {
            "similarity": avg_similarity,
            "agreement": agreement
        }

    def verify(self, predicted_class, x_input, confidence):
        similarities = []

        for i in range(len(self.X_train)):
            sim = self._compute_similarity(x_input, self.X_train[i])
            similarities.append((sim, self.y_train[i]))

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[:self.k]

        sims = [s for s, _ in top_k]
        labels = [l for _, l in top_k]

        avg_similarity = np.mean(sims)

        agreement = np.mean([1 if l == predicted_class else 0 for l in labels])

        num_symptoms = int(np.sum(x_input))

        if num_symptoms < 4:
            sim_threshold = 0.4
        elif num_symptoms < 7:
            sim_threshold = 0.5
        else:
            sim_threshold = self.thresholds["similarity"]

        agreement_threshold = self.thresholds["agreement"]
        confidence_threshold = self.thresholds["confidence"]

        info = {
            "similarity": avg_similarity,
            "agreement": agreement,
            "confidence": confidence,
            "reason": ""
        }

        if avg_similarity >= sim_threshold and agreement >= agreement_threshold:
            info["reason"] = "High similarity and agreement"
            return "accept", info

        if confidence >= confidence_threshold:
            info["reason"] = "Accepted due to high model confidence"
            return "accept", info

        if avg_similarity < sim_threshold:
            info["reason"] = f"Low similarity ({avg_similarity:.2f})"
            return "uncertain", info

        if agreement < agreement_threshold:
            counts = {}
            for l in labels:
                counts[l] = counts.get(l, 0) + 1

            suggested = max(counts, key=counts.get)

            info["reason"] = f"Low agreement ({agreement:.2f})"
            info["suggested"] = suggested

            return "reject", info

        info["reason"] = "Borderline case"
        return "uncertain", info