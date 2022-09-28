import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inline
import matplotlib

sales = pd.read_csv(
    'C:/Users/darla/Desktop/notebook/FreeCodeCamp-Pandas-Real-Life-Example-master/data/sales_data.csv',
    parse_dates=['Date'])
sales.head()

sales['Customer_Age'].mean()

sales['Customer_Age'].plot(kind='box', vert=False, figsize=(14,6))

sales['Order_Quantity'].mean()
sales['Order_Quantity'].plot(kind='hist', bins=30, figsize=(14,6))
sales['Order_Quantity'].plot(kind='box', vert=False, figsize=(14,6))

sales['Year'].value_counts()
sales['Year'].value_counts().plot(kind='pie', figsize=(6,6))

sales['Country'].value_counts().head(1)
sales['Country'].value_counts()
sales['Country'].value_counts().plot(kind='bar', figsize=(14,6))

sales['Product'].unique()
sales['Product'].value_counts().head(10).plot(kind='bar', figsize=(14,6))

sales.plot(kind='scatter', x='Unit_Cost', y='Unit_Price', figsize=(6,6))
sales.plot(kind='scatter', x='Order_Quantity', y='Profit', figsize=(6,6))

sales[['Profit', 'Country']].boxplot(by='Country', figsize=(10,6))

sales[['Customer_Age', 'Country']].boxplot(by='Country', figsize=(10,6))

sales['Calculated_Date'] = sales[['Year', 'Month', 'Day']].apply(lambda x: '{}-{}-{}'.format(x[0], x[1], x[2]), axis=1)
sales['Calculated_Date'].head()

sales['Calculated_Date'] = pd.to_datetime(sales['Calculated_Date'])
sales['Calculated_Date'].head()

sales['Calculated_Date'].value_counts().plot(kind='line', figsize=(14,6))

sales['Revenue'] += 50

sales['Calculated_Date'] = sales[['Year', 'Month', 'Day']].apply(lambda x: '{}-{}-{}'.format(x[0], x[1], x[2]), axis=1)
sales['Calculated_Date'].head()

france_states = sales.loc[sales['Country'] == 'France', 'State'].value_counts()

france_states

france_states.plot(kind='bar', figsize=(14,6))

sales['Product_Category'].value_counts()


sales['Product_Category'].value_counts().plot(kind='pie', figsize=(6,6))