import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)

A[0, 1]

A[0:2, 0:2] = 10
print(A)

A = np.array([[1, 2, 3], [4, 5, 6]])

print(A<5)

print(A[A < 5])

A[A<5] = 4
print(A)