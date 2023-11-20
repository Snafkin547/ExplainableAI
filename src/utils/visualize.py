import matplotlib.pyplot as plt


def visualize(history):
    acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    print("Accuracy of the Model: {}% \n Error: {}".format(acc[-1] * 100, loss[-1]))

    plt.subplot(2, 1, 1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "y", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.subplot(2, 1, 2)
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "y", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
