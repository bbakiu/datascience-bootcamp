import matplotlib.pyplot as plt
import pandas as pd

df3 = pd.read_csv('df3.csv')
print(df3.head())

df3.plot.scatter(x='a', y='b', c='red', s=50, figsize=(12, 3))
plt.show()

df3['a'].hist(bins=10)
plt.show()

plt.style.use('ggplot')

df3['a'].plot.hist(alpha=0.5, bins=25)
plt.show()

df3[['a', 'b']].plot.box()
plt.show()

df3['d'].plot.kde()
plt.show()

df3.head(30).plot.area()
plt.show()
