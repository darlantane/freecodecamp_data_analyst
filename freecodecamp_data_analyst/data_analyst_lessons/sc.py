import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 10)
y = np.sin(x)
plt.scatter(x, y)

from scipy.interpolate import interp1d

f = interp1d(x, y, kind='cubic')

new_x = np.linspace(0, 10, 50)
result = f(new_x)

plt.scatter(x, y)
plt.plot(new_x, result, c='r')

x = np.linspace(0, 2, 100)
y = 1/3*x**3 - 3/5 * x**2 + 2 + np.random.randn(x.shape[0])/20
plt.scatter(x, y)

def f (x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


from scipy import optimize
params, param_cov = optimize.curve_fit(f, x, y)
plt.scatter(x, y)
plt.plot(x, f(x, params[0], params[1], params[2], params[3]), c='g', lw=3)