import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('Ecommerce Customers')

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df)
plt.show()

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df)
plt.show()

sns.jointplot(x='Time on App', y='Length of Membership', data=df,kind="hex")
plt.show()

sns.pairplot(df)
plt.show()

sns.lmplot(x='Yearly Amount Spent', y='Length of Membership', data=df.reset_index())
plt.show()

