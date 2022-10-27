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