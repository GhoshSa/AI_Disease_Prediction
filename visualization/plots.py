import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import entropy

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
    plt.scatter(np.max(probs, axis=1), -np.sum(probs * np.log(probs + 1e-9), axis=1), alpha=0.5)
    plt.xlabel("Max Confidence")
    plt.ylabel("Entropy")
    plt.title("Confidence vs Entropy")
    plt.savefig("confidence_vs_entropy.png", dpi=300)
    plt.close()

def plot_entropy_distribution(y, probs):
    pred = np.argmax(probs, axis=1)
    ent = entropy(probs)
    plt.hist(ent[pred == y], bins=30, alpha=0.7, label="Correct")
    plt.hist(ent[pred != y], bins=30, alpha=0.7, label="Incorrect")
    plt.legend()
    plt.xlabel("Entropy")
    plt.title("Entropy Distribution")
    plt.savefig("entropy_distribution.png", dpi=300)
    plt.close()