import numpy as np
import pandas as pd
from NeuralLayer import Layer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# TODO: change the way that the layer works so there is no need for non and non_deriv
def non(x):
    return x

def non_deriv(x):
    return 1


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

X = data.drop('quality', axis=1).values
y = data['quality'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x = np.array(X_train)
y = np.array(y_train).reshape(-1,1)
print(y)
X_test = np.array(X_test)
y_test = np.array(y_test)

layer1 = Layer(11 , 30, sigmoid, sigmoid_derivative)
layer2 = Layer(30, 1, non, non_deriv)

layer1.forward(x)
layer2.forward(layer1.predY)
loss = np.mean((layer2.predY - y) ** 2)
print(loss)

for i in range(1000):
    layer1.forward(x)
    layer2.forward(layer1.predY)
    gradX2, gradB2, gradW2 = layer2.backward(layer1.predY, 2 * (layer2.predY - y))

    layer2.W1 -= gradW2 * 0.0001
    layer2.b1 -= gradB2 * 0.0001

    gradX1, gradB1, gradW1 = layer1.backward(x, gradX2)

    layer1.W1 -= gradW1 * 0.0001
    layer1.b1 -= gradB1 * 0.0001


layer1.forward(x)
layer2.forward(layer1.predY)
loss = np.mean((layer2.predY - y) ** 2)
print(loss)