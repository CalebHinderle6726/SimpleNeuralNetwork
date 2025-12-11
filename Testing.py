import numpy as np
import matplotlib.pyplot as plt
from NeuralLayer import Layer
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

np.random.seed(42)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def identity(x):
    return x


def identity_derivative(x):
    return np.ones_like(x)


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(probs, y_true_one_hot):
    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)    # preventing inf from log(0)
    return -np.mean(np.sum(y_true_one_hot * np.log(probs), axis=1))


def one_hot(y, num_classes):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out


def accuracy(probs, y_true_labels):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_true_labels)


# Loading MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0  # scale to [0,1]
y = mnist.target.astype(int)

# Standardize for more stable training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratify so no class imbalance
xTrain, xTest, yTrain_labels, yTest_labels = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

yTrain = one_hot(yTrain_labels, num_classes=10)
yTest = one_hot(yTest_labels, num_classes=10)

# 784 x 256 x 128 x 10 logits
layer1 = Layer(784, 256, relu, relu_derivative)
layer2 = Layer(256, 128, relu, relu_derivative)
layer3 = Layer(128, 10, identity, identity_derivative)

nn = NeuralNetwork([layer1, layer2, layer3])

epochs = 15
batch_size = 128
learning_rate = 0.05

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

num_batches = int(np.ceil(xTrain.shape[0] / batch_size))

for epoch in range(epochs):
    # Shuffling each epoch
    indices = np.random.permutation(xTrain.shape[0])
    xTrain_shuffled = xTrain[indices]
    yTrain_shuffled = yTrain[indices]

    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        xb = xTrain_shuffled[start:end]
        yb = yTrain_shuffled[start:end]

        logits = nn.forward(xb)
        probs = softmax(logits)
        grad_logits = (probs - yb) / xb.shape[0]
        nn.backward(xb, grad_logits, learning_rate)

    # Saving metrics after epoch
    train_probs = softmax(nn.forward(xTrain))
    test_probs = softmax(nn.forward(xTest))

    train_loss = cross_entropy(train_probs, yTrain)
    test_loss = cross_entropy(test_probs, yTest)
    train_acc = accuracy(train_probs, yTrain_labels)
    test_acc = accuracy(test_probs, yTest_labels)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"train loss: {train_loss:.4f} - test loss: {test_loss:.4f} - "
        f"train acc: {train_acc:.4f} - test acc: {test_acc:.4f}"
    )

# Plotting losses
plt.style.use('dark_background')
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label="Train loss")
plt.plot(range(1, epochs + 1), test_losses, label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy")
plt.title("Loss over epochs")
plt.legend()

# Plotting accuracy
plt.style.use('dark_background')
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label="Train accuracy")
plt.plot(range(1, epochs + 1), test_accuracies, label="Test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over epochs")
plt.legend()

plt.savefig('plot.png')
