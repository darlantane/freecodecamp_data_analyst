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