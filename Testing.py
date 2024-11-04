import numpy as np
import pandas as pd
from NeuralLayer import Layer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import NeuralNetwork

# Setting up training data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

X = data.drop('quality', axis=1).values
y = data['quality'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, random_state=42)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain).reshape(-1,1)
xTest = np.array(xTest)
yTest = np.array(yTest)


# Training and testing
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

layer1 = Layer(11 , 30, sigmoid, sigmoid_derivative)
layer2 = Layer(30, 1)

nn = NeuralNetwork(np.array([layer1, layer2]))
print(nn.loss(xTrain, yTrain))
print(nn.loss(xTest, yTest))

nn.train(xTrain, yTrain, 1000, 0.0001)

print(nn.loss(xTrain, yTrain))
print(nn.loss(xTest, yTest))