import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
import seaborn as sns

titanic = sns.load_dataset('titanic')
titanic

y = titanic['survived']
X = titanic.drop('survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)