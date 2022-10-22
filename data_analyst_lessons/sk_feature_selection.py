import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

plt.plot(X)
plt.legend(iris.feature_names)

X.var(axis=0)

selector = VarianceThreshold(threshold=0.2)
selector.fit(X)

selector.get_support()

np.array(iris.feature_names)[selector.get_support()]

selector.variances_

from sklearn.feature_selection import SelectKBest, chi2, f_classif
chi2(X, y)

selector = SelectKBest(f_classif, k=2)
selector.fit(X, y)
selector.scores_

np.array(iris.feature_names)[selector.get_support()]
