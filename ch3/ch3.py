import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim  as optim
import numpy as np

# data = pd.read_csv('hour.csv')
# x = data.instant[:50]
# y = data.cnt[:50]
# # print(data.head())
# plt.figure(figsize=(10,10))
# plt.plot(x,y)
# plt.plot(x,y,'o',c='r')
# plt.show()

dataPath = 'hour.csv'
rides = pd.read_csv(dataPath)
counts = rides.cnt[:50]
x = np.arange(len(counts))
y = np.array(counts)
plt.figure(figsize=(10,7))
plt.plot(x,y,'o-')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

x = torch.tensor(np.arange(len(counts),dtype=float)/len(counts),requires_grad=True)
y = torch.tensor(np.array(counts,dtype=float),requires_grad=True)

sz = 10
weights = torch.randn((1,sz),dtype=torch.double,requires_grad=True)
biases = torch.randn(sz,dtype=torch.double,requires_grad=True)
weights2 = torch.randn((sz,1),dtype=torch.double,requires_grad=True)
learningRate = 0.001
losses = []
x = x.view(50, -1)
y = y.view(50, -1)
for i in range(100000):
    hidden = x * weights + biases
    hidden = torch.sigmoid(hidden)
    predictions = hidden.mm(weights2)
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())
    if(i % 10000 == 0):
        print('loss', i, ': ', loss)
    loss.backward()
    weights.data.add_(-learningRate * weights.grad.data)
    biases.data.add_(-learningRate * biases.grad.data)
    weights2.data.add_(-learningRate * weights2.grad.data)
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()

plt.plot(losses)
plt.show()

x_data = x.data.numpy()
plt.figure(figsize=(10,7))
plt.plot(x_data, y.data.numpy(),'o')
plt.plot(x_data, predictions.data.numpy())
plt.show()

counts_predict = rides.cnt[50:100]
x = torch.tensor(np.arange(50,100, dtype=float) / len(counts),dtype=float,requires_grad=True)
y = torch.tensor(np.array(counts_predict, dtype=float),requires_grad=True)
x = x.view(50,-1)
y = y.view(50,-1)

hidden = x * weights + biases
hidden = torch.sigmoid(hidden)
predictions = hidden.mm(weights2)
loss = torch.mean((predictions - y) ** 2)
print(loss)
x_data = x.data.numpy()
plt.figure(figsize=(10,7))
plt.plot(x_data,y.data.numpy(),'o')
plt.plot(x_data,predictions.data.numpy())
plt.show()
