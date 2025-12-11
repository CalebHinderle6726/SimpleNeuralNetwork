import numpy as np

# This NeuralNetwork class defines the network as a array of layers and handles learning and backprop
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = list(layers)
        self.reverseLayers = list(reversed(self.layers))
    
    def forward(self, X):
        prevOut = X
        for layer in self.layers:
            layer.forward(prevOut)
            prevOut = layer.predY
        return self.layers[-1].predY

    def backward(self, X, grad, learningRate):
        for i in range(len(self.reverseLayers)):
            if i + 1 < len(self.reverseLayers):
                grads = self.reverseLayers[i].backward(self.reverseLayers[i+1].predY, grad)
            else:
                grads = self.reverseLayers[i].backward(X, grad)
            self.reverseLayers[i].W1 -= learningRate * grads[2]
            self.reverseLayers[i].b1 -= learningRate * grads[1]
            grad = grads[0]

    def train(self, X, y, epochs, learningRate):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y, learningRate)

    def loss(self, X, y):
        return np.mean((self.forward(X) - y.reshape(-1, 1)) ** 2)
