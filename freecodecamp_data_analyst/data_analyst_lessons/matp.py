import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 2, 10)
y = X**2

plt.plot(X, y)
plt.show()

plt.scatter(X, y)
plt.show()

X = np.linspace(0, 2, 10)

plt.figure()
plt.plot(X, X**2, label='quadratique')
plt.plot(X, X**3, label='cubique')

plt.title('figure 1')
plt.xlabel('axe x')
plt.ylabel('axe y')
plt.legend()

plt.savefig('figure.png')
plt.show()

plt.subplot(2, 1, 1)
plt.plot(X, y, c='red')
plt.subplot(2, 1, 2)
plt.plot(X, y, c='blue')

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(X, y)
ax[1].plot(X, np.sin(X))
plt.show()

