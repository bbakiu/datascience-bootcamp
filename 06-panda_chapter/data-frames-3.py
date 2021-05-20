import pandas as pd
from numpy.random import randn

outside = 'G1 G1 G1 G2 G2 G2'.split()
print(outside)
inside = [1, 2, 3, 1, 2, 3]

hier_index = list(zip(outside, inside))
print(hier_index)
hier_index = pd.MultiIndex.from_tuples(hier_index)
print(hier_index)

df = pd.DataFrame(randn(6, 2), hier_index, ['A', 'B'])
print(df)

print(df.loc['G1'].loc[1])
df.index.names = ["Groups", 'Num']
print(df)
print(df.loc['G2'].loc[2]['B'])
print(df.loc['G1'].loc[3]['A'])
print(df.xs('G1'))

print(df.xs(1, level='Num'))
