import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('C:/Users/darla/PycharmProjects/freecodecamp_data_analyst/data_analyst_lessons/titanic.xls')
data.shape
data.head()

data = data.drop(['name', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1)

data = data.dropna(axis=0)
data.shape

data['age'].hist()

data.groupby(['sex']).mean()

data['pclass'].value_counts()

data[data['age'] < 18]['pclass'].value_counts()

def category_ages(age):
    if age <= 20:
        return '<20 ans'
    elif (age > 20) & (age <= 30):
        return '20-30 ans'
    elif (age > 30) & (age <= 40):
        return '30-40 ans'
    else:
        return '+40 ans'

data['age'] = data['age'].map(category_ages)
data['sex'].astype('category').cat.codes



