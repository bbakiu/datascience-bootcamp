import pandas as pd

df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [444, 555, 666, 444], 'col3': ['abc', 'def', 'ghi', 'xyz']})
df.head()
print(df)

print(df['col2'].unique())
print(df['col2'].nunique())
print(df['col2'].value_counts())

print(df[df['col1'] > 2])
print(df['col1'] > 2)


def times2(x):
    return x * 2


print(df['col3'].apply(times2))
print(df['col3'].apply(len))

print(df['col2'].apply(lambda x: x * 2))

print(df.drop('col1', axis=1))
print(df.columns)
print(df.index)

print(df.sort_values(by='col2'))

print(df.isnull())

data = {'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
        'B': ['one', 'one', 'two', 'two', 'one', 'one'],
        'C': ['x', 'y', 'x', 'y', 'x', 'y'],
        'D': [1, 3, 2, 5, 4, 1]}

df2 = pd.DataFrame(data)
print(df2)

print(df2.pivot_table(values='D', index=['A', 'B'], columns=['C']))
