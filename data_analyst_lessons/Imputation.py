import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier

from sklearn.impute import SimpleImputer
X = np.array([[10, 3],
              [0, 4],
              [5, 3],
              [np.nan, 3]])

imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean')

imputer.fit_transform(X)

X_test = np.array([[12, 5],
                   [40, 2],
                   [5, 5],
                   [np.nan, np.nan]])

imputer.transform(X_test)

from sklearn.impute import KNNImputer
X = np.array([[1, 100],
              [2, 30],
              [3, 15],
              [np.nan, 20]])

imputer = KNNImputer(n_neighbors=1)
imputer.fit_transform(X)

X_test = np.array([[np.nan, 35]])

imputer.transform(X_test)

from sklearn.impute import MissingIndicator
from sklearn.pipeline import make_union
X = np.array([[1, 100],
              [2, 30],
              [3, 15],
              [np.nan, np.nan]])

MissingIndicator().fit_transform(X)

pipeline = make_union(SimpleImputer(strategy='constant', fill_value=-99),
                      MissingIndicator())

pipeline.fit_transform(X)