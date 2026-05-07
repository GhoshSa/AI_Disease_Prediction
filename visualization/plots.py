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

def plot_confusion_matrix(cm, filename="confusion_matrix.png", title="Confusion Matrix", class_names=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')

    plt.title(title, fontsize=16, pad=15)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)

    num_classes = cm.shape[0]

    ticks = np.arange(num_classes)

    if class_names is not None:
        plt.xticks(ticks, class_names, rotation=45, ha="right", fontsize=8)
        plt.yticks(ticks, class_names, fontsize=8)
    else:
        plt.xticks(ticks)
        plt.yticks(ticks)

    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)

    plt.grid(False)

    cm_display = cm

    thresh = cm_display.max() / 2

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_display[i, j]

            plt.text(
                j, i,
                f"{value:.0f}" if cm_display is cm else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if value > thresh else "black"
            )

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_clean_vs_noisy(clean_metrics, noisy_metrics):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    clean_vals = [
        clean_metrics['accuracy'],
        clean_metrics['precision'],
        clean_metrics['recall'],
        clean_metrics['f1']
    ]

    noisy_vals = [
        noisy_metrics['accuracy'],
        noisy_metrics['precision'],
        noisy_metrics['recall'],
        noisy_metrics['f1']
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))

    plt.bar(x - width/2, clean_vals, width, label='Clean Data')
    plt.bar(x + width/2, noisy_vals, width, label='Noisy Data')

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Performance Comparison: Clean vs Noisy Data")
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics_comparison.png", dpi=300)
    plt.close()