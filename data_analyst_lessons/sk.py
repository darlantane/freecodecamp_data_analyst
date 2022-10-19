import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
m = 100
X = np.linspace(0, 10, m).reshape(m,1)
y = X + np.random.randn(m, 1)

plt.scatter(X, y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
model.score(X, y)

plt.scatter(X, y)
plt.plot(X, model.predict(X), c='red')

titanic = sns.load_dataset('titanic')
titanic = titanic[['survived', 'pclass', 'sex', 'age']]
titanic.dropna(axis=0, inplace=True)
titanic['sex'].replace(['male', 'female'], [0, 1], inplace=True)
titanic.head()

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
y = titanic['survived']
X = titanic.drop('survived', axis=1)
model.fit(X, y)
model.score(X, y)

def survie(model, pclass=3, sex=0, age=26):
    x = np.array([pclass, sex, age]).reshape(1, 3)
    print(model.predict(x))
    print(model.predict_proba(x))
survie(model)
[0]