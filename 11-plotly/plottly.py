import cufflinks as cf
import numpy as np
import pandas as pd
from plotly import __version__
from plotly.offline import init_notebook_mode
import matplotlib.pyplot as plt

print(__version__)
init_notebook_mode(connected=True)

cf.go_offline()

df = pd.DataFrame(np.random.randn(100, 4), columns='A B C D'.split())
df.head()

df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': ['32', '43', '50']})
df2.head()

df.iplot()
plt.show()

df.plot()
plt.show()

df.iplot(kind='scatter', x='A', y='B', mode='markers')
plt.show()

df2.iplot(kind='bar', x='Category', y='Values')
plt.show()


df.sum().iplot(kind='bar')
plt.show()

df.iplot(kind='box')
plt.show()

df3 = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50], 'y': [500, 400, 300, 200, 100]})

print(df3)
df3.iplot(kind='surface', colorscale='rdylbu')
plt.show()

df['A'].iplot(kind='hist', bins=25)
plt.show()


