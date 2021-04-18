import numpy as np

my_list = [1, 2, 3]
arr = np.array(my_list)
print(arr)

my_mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

arr2d = np.array(my_mat)
print(arr2d)

# quick generate array
print(np.arange(0, 11, 2))

print(np.zeros(3))

print(np.zeros((2, 3)))

print(np.ones(10))

print(np.linspace(0, 5, 100))

print(np.eye(4))

print(np.random.rand(5))

print(np.random.rand(5, 6))

print(np.random.randn(5))

print(np.random.randint(0, 100, 8))

np_array = np.arange(25)

np_random_array = np.random.randint(0, 50, 10)

print(np_array.reshape(5, 5))

print(np_random_array.max())
print(np_random_array.min())

print(np_random_array.argmax())
print(np_random_array.argmin())

print(np_random_array.shape)

print(np_random_array.dtype)