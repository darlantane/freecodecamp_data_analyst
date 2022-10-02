import numpy as np

A = np.array([1, 2, 3])
A = np.zeros((2, 3))
B = np.ones((2, 3))
C = np.random.randn(2, 3)
D = np.random.rand(2, 3)
E = np.random.randint(0, 10, [2, 3])

A = np.ones((2, 3), dtype=np.float16)

A = np.linspace(1,10, 10)
B = np.arange(0, 10, 10)

A = np.zeros((2, 3))

print(A.size)
print(A.shape)

print(type(A.shape))

print(A.shape[0])