import matplotlib.pyplot as plt
import json

def plot_training_metrics(log_file="training_log.json"):
    # load logged metrics
    with open(log_file, "r") as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-o', label="Train Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], 'g-o', label="Train Acc")
    plt.plot(epochs, history["val_acc"], 'r-o', label="Val Acc")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_metrics()
