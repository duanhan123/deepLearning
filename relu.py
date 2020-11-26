#%%
import matplotlib.pyplot as plt
import numpy as np

g = lambda z:(np.maximum(0, z))
start = -10
stop = 10
step = 0.01
num = int((stop - start) / step)
x = np.linspace(start, stop, num)
y = g(x)

fig = plt.figure(figsize=(10, 7))
plt.plot(x, y, label='relu')
plt.legend()
plt.grid()
plt.show()