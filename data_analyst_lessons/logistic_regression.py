import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))
plt.scatter(X[:,0], X[:, 1], c=y, cmap='bwr')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)
def forward_propagation(X, W, b):
    Z = X.dot(W) + b
    A = sigmoid(Z)
    return A

def log_loss(y, A):
    return 1/len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
def gradients(X, A, y):
    dW = 1/len(y) * np.dot(X.T, A - y)
    db = 1/len(y) * np.sum(A - y)
    return (dW, db)

def optimisation(X, W, b, A, y, learning_rate):
    dW, db = gradients(X, A, y)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)


def predict(X, W, b):
    A = forward_propagation(X, W, b)
    return A >= 0.5

def visualisation(X, y, W, b):
    resolution = 300
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x1 = np.linspace(xlim[0], xlim[1], resolution)
    x2 = np.linspace(ylim[0], ylim[1], resolution)
    X1, X2 = np.meshgrid(x1, x2)

    XX = np.vstack((X1.ravel(), X2.ravel())).T

    Z = predict(XX, W, b)
    Z = Z.reshape((resolution, resolution))

    ax.pcolormesh(X1, X2, Z, zorder=0, alpha=0.1)
    ax.contour(X1, X2, Z, colors='g')