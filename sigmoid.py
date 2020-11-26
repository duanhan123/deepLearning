#%%
import matplotlib.pyplot as plt
import numpy as np

g = lambda z:1/(1+np.exp(-z))
start = -10
stop = 10
step = 0.01
num = (stop - start) / step
x = np.linspace(start, stop, int(num))
# print(x)
y = g(x)
fig = plt.figure(figsize=(10,7))
plt.plot(x, y, label='sigmoid')
plt.grid(True)
plt.legend()
plt.show()
