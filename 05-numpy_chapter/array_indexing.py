import numpy as np

arr = np.arange(0, 11)
print(arr)

print(arr[8])
print(arr[1:5])

print(arr[:6])

print(arr[-4:-1])

print(arr[-1])
slice_arr = arr[0:6]
arr[0:5] = 100
print(arr, slice_arr)

arr_copy = arr.copy()

arr_2d = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print(arr_2d)

print(arr_2d[2, 0])
print(arr_2d[2][0])

print(arr_2d[:2, 1:])

# Conditional array
arr_1d = np.arange(1, 11)

print(arr_1d > 5)

bool_arr = arr_1d > 5

print(arr_1d[bool_arr])

print(arr_1d[arr_1d > 5])

print(arr_1d[arr_1d < 5])
