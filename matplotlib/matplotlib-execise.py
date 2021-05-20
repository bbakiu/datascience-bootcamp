import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 100)
y = x * 2
z = x ** 2

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y)
plt.show()

fig2 = plt.figure()
ax1 = fig2.add_axes([0, 0, 1, 1])
ax2 = fig2.add_axes([0.2, 0.5, .2, .2])
ax1.plot(x, y, 'r')
ax2.plot(x, y, 'r')
plt.show()

fig3 = plt.figure()
ax31 = fig3.add_axes([0, 0, 1, 1])
ax32 = fig3.add_axes([0.2, 0.5, .4, .4])
ax32.set_xlim([20, 22])
ax32.set_ylim([30, 50])
ax32.plot(x, y, 'r')
ax31.plot(x, z, 'r')
plt.show()

fig4, axes = plt.subplots(nrows=1, ncols=2, figsize=([8, 2]))
axes[0].plot(x, y, 'b--')
axes[1].plot(x, z, 'r', lw=5)
plt.show()
