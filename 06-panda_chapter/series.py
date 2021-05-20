import numpy as np
import pandas as pd

labels = ['a', 'b', 'c']
my_data = [10, 20, 30]

arr = np.array(my_data)
d = {'a': 10, 'b': 20, 'c': 30}

series = pd.Series(data=my_data)
print(series)
series = pd.Series(data=my_data, index=labels)
print(series)

series = pd.Series(arr)
print(series)

series = pd.Series(d)
print(series)

ser1 = pd.Series([1, 2, 3, 4], ['USA', 'Germany', 'USSR', 'Japan'])
print(ser1)

ser2 = pd.Series([1, 2, 5, 4], ['USA', 'Germany', 'Italy', 'Japan'])
print(ser2)

print(ser1['USA'])

ser3 = pd.Series(data=labels)
print(ser3[0])

print(ser1+ser2)
