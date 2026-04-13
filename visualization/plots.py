import matplotlib.pyplot as plt

def plot_training_loss(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Progression', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_loss.png", dpi=300)
    plt.close()

def plot_training_accuracy(history):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_acc'], 'g-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'm--', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy Progression', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_accuracy.png", dpi=300)
    plt.close()