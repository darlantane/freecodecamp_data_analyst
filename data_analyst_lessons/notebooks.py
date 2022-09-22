import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sales = pd.read_csv(
    'C:/Users/darla/Desktop/notebook/FreeCodeCamp-Pandas-Real-Life-Example-master/data/sales_data.csv',
    parse_dates=['Date'])
sales.head()