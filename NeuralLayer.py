import numpy as np

class Layer:
    def __init__(self, inputSize, outputSize, activation, derivative):
        # Input and output of the current layer
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.activation = activation
        self.derivative = derivative

        # init Weights of the current layer
        self.W1 = np.random.randn(self.inputSize, self.outputSize) / np.sqrt(self.inputSize)
        self.b1 = np.zeros((1, self.outputSize))

    def forward(self, X):
        # The predicted output for this layer
        predY = np.dot(X, self.W1) + self.b1

        # Non-linear activation
        predY = self.activation(predY)

        # Storing layers predicted output
        self.predY = predY

    def backward(self, X, grad):
        # The gradient of the error wrt. the output of the current layer 
        gradWrtActiv = grad * self.derivative(self.predY)

        # Gradients of the error wrt. the current layers w, b, and input values
        gradW = np.dot(X.T, gradWrtActiv)
        gradB = np.sum(gradWrtActiv, axis=0, keepdims=True)
        gradX = np.dot(gradWrtActiv, self.W1.T)

        return gradX, gradB, gradW
