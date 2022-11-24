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

def model(X, theta):
    return X.dot(theta)

plt.scatter(x, y)
plt.scatter(x, model(X, theta), c='r')

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)
cost_function(X, y, theta)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)
def gradient_descent(X, y, theta, learning_rate, n_iterations):

    cost_history = np.zeros(n_iterations)

    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)

    return theta, cost_history

n_iterations = 1000
learning_rate = 0.01

theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
theta_final

predictions = model(X, theta_final)

plt.scatter(x, y)
plt.scatter(x, predictions, c='r')

plt.plot(range(n_iterations), cost_history)

def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v
coef_determination(y, predictions)

np.random.seed(0)

x, y = make_regression(n_samples=100, n_features=2, noise = 10)

plt.scatter(x[:,0], y)

from mpl_toolkits.mplot3d import Axes3D
#%matplotlib notebook #activez cette ligne pour manipuler le graph 3D

ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[:,0], x[:,1], y)

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('y')

# Verification des dimensions
print(x.shape)
print(y.shape)

y = y.reshape(y.shape[0], 1)
print(y.shape)

X = np.hstack((x, np.ones((x.shape[0], 1)))) # ajoute un vecteur Biais de dimension (x.shape[0], 1)

print(X.shape)
print(X[:10])

theta = np.random.randn(3, 1)
theta


n_iterations = 1000
learning_rate = 0.01

theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
predictions = model(X, theta_final)

theta_final

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[:,0], x[:,1], y)
ax.scatter(x[:,0], x[:,1], predictions)

plt.plot(range(n_iterations), cost_history)

coef_determination(y, predictions)
