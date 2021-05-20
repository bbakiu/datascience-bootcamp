import numpy as np

# 1
print(np.zeros(10))

# 2
print(np.ones(10))

# 3
print(np.ones(10) * 5)

# 4
print(np.arange(10, 51))

# 5
print(np.arange(9).reshape((3, 3)))

# 6
print(np.eye(3, 3))

# 7
print(np.random.rand(1))

# 8
print(np.random.randn(25))

# 9
print(np.linspace(0.01, 1., 100).reshape(10, 10))

# 10
print(np.linspace(0, 1, 20))

# 11
mat = np.arange(1, 26).reshape(5, 5)
print(mat[2:, 1:])

# 12
print(mat[3, 4])

# 13
print(mat[:3, [1]])

# 14
print(mat[4, :])

# 15
print(mat[3:, :])

# 16
print(mat.sum())

# 17
print(mat.std())

# 18
print(mat.sum(axis=0))
