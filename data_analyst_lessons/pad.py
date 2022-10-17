import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('titanic.xls')
data.shape
data.head()

data = data.drop(['name', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1)

data = data.dropna(axis=0)
data.shape

data['age'].hist()

data.groupby(['sex']).mean()

data['pclass'].value_counts()

data[data['age'] < 18]['pclass'].value_counts()



