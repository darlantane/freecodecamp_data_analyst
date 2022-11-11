import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

np.random.seed(0)

x, y = make_regression(n_samples=100, n_features=1, noise = 10)
y = y + abs(y/2)

plt.scatter(x, y)

print(x.shape)
print(y.shape)

y = y.reshape(y.shape[0], 1)
print(y.shape)

X = np.hstack((x, np.ones(x.shape)))
X = np.hstack((x**2, X))

print(X.shape)
print(X[:10])

theta = np.random.randn(3, 1)
theta