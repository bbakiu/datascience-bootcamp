import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df1 = pd.read_csv('df1.csv', index_col=0)
df2 = pd.read_csv('df2.csv')
print(df1.head())
print(df2.head())

df1['A'].hist(bins=30)
plt.show()

df1['A'].plot(kind='hist')
plt.show()

df1['A'].plot.hist()
plt.show()

df2.plot.area(alpha=.2)
plt.show()

df2.plot.bar()
plt.show()

df1['A'].plot.hist()
plt.show()

df1.plot.line(y='B', figsize=(12, 3))
plt.show()

df1.plot.scatter(x='A', y='B', s=df1['C'] * 100)
plt.show()

df2.plot.box()
plt.show()

df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
print(df.head())

df.plot.hexbin(x='a', y='b', gridsize=25)
plt.show()

df2['a'].plot.kde()
df2.plot.density()
plt.show()