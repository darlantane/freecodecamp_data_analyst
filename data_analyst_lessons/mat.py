import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])

print(A.sum())
print(A.sum(axis=0))
print(A.sum(axis=1))
print(A.cumsum(axis=0))

print(A.prod())
print(A.cumprod())

print(A.min())
print(A.max())

print(A.mean())
print(A.std())
print(A.var())

A = np.random.randint(0, 10, [5, 5]) # tableau aléatoire
print(A)

print(A.argsort())
print(A[:,0].argsort())