import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)

df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

print(df)

print(df > 0)

booldf = df > 0
print(df[booldf])
print(df[df > 0])

print(df['W'] > 0)
print(df[df['W'] > 0])

print(df['Z'] < 0)

print(df[df['Z'] < 0])

result_df = df[df['W'] > 0]

print(result_df['X'])
print(df[df['W'] > 0]['X'])

print(df[(df['W'] > 0) | (df['Y'] > 1)])

print(df)
print(df.reset_index())
newind = 'CA NY WY OR CO'.split()
print(df)
df['States'] = newind
print(df)
print(df.set_index('States'))
