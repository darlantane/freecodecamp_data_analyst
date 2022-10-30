import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/MachineLearnia/Python-Machine-Learning/master/Dataset/dataset.csv'
data = pd.read_csv(url, index_col=0, encoding = "ISO-8859-1")
data.head()

df = data.copy()
df.head()

missing_rate = df.isna().sum()/df.shape[0]
blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate >0.88)])
viral_columns = list(df.columns[(missing_rate < 0.80) & (missing_rate > 0.75)])
key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']
df = df[key_columns + blood_columns + viral_columns]
df.head()

from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(df, test_size=0.2, random_state=0)
trainset['SARS-Cov-2 exam result'].value_counts()

testset['SARS-Cov-2 exam result'].value_counts()

def encodage(df):
    code = {'negative':0,
            'positive':1,
            'not_detected':0,
            'detected':1}

    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)

    return df

def feature_engineering(df):
    df['est malade'] = df[viral_columns].sum(axis=1) >= 1
    df = df.drop(viral_columns, axis=1)
    return df
def imputation(df):
    #df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leukocytes'].isna())
    #df = df.fillna(-999)
    df = df.dropna(axis=0)
    return  df
def preprocessing(df):

    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)

    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']

    print(y.value_counts())

    return X, y
X_train, y_train = preprocessing(trainset)

X_test, y_test = preprocessing(testset)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

model_1 = RandomForestClassifier(random_state=0)
model_2 = make_pipeline(PolynomialFeatures(2), SelectKBest(f_classif, k=10),
                        RandomForestClassifier(random_state=0))

