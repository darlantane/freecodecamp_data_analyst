import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)
url = 'https://raw.githubusercontent.com/MachineLearnia/Python-Machine-Learning/master/Dataset/dataset.csv'
data = pd.read_csv(url, index_col=0, encoding = "ISO-8859-1")

data.head()

df = data.copy()
df.shape

df.dtypes.value_counts().plot.pie()

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)

(df.isna().sum()/df.shape[0]).sort_values(ascending=True)

df = df[df.columns[df.isna().sum()/df.shape[0] <0.9]]
df.head()

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)

for col in df.select_dtypes('float'):
    plt.figure()
    sns.distplot(df[col])

for col in df.select_dtypes('object'):
    print(f'{col :-<50} {df[col].unique()}')

for col in df.select_dtypes('object'):
    plt.figure()
    df[col].value_counts().plot.pie()

positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']
negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']

missing_rate = df.isna().sum()/df.shape[0]
blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate >0.88)]
viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]

for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()

sns.countplot(x='Patient age quantile', hue='SARS-Cov-2 exam result', data=df)

pd.crosstab(df['SARS-Cov-2 exam result'], df['Influenza A'])

for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')