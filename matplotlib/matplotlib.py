import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline

x = np.linspace(0, 5, 11)
y = x ** 2
print(x, y)
plt.plot(x, y, 'r-')
plt.show()

plt.plot(x, y, 'r-')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()

plt.subplot(1, 2, 1)
plt.plot(x, y, 'r')

plt.subplot(1, 2, 2)
plt.plot(y, x, 'b')
plt.show()

# Object oriened methods
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y, 'g')
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
plt.show()

fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.6, 0.4, 0.3])

axes1.plot(x, y)
axes2.plot(y, x)
plt.show()

#### Part 2

fig, axes = plt.subplots(nrows=1, ncols=2)
# # axes.plot(x,y)
# for current_ax in axes:
#     current_ax.plot(x, y)

axes[0].plot(x, y)
axes[1].plot(y, x)
plt.show()

# fig = plt.figure(figsize=(8, 2), dpi=100)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 2))
axes[0].plot(x, y)
axes[1].plot(y, x)
plt.tight_layout()
plt.show()

fig.savefig('my_pic.png')

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, x ** 2, label='X Squared')
ax.plot(x, x ** 3, label='X Cubed')

ax.legend(loc='best')
plt.show()
