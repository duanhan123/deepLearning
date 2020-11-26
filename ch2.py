import torch
import numpy as np
x = torch.rand(5,3)
# print(x)
y = torch.ones(5,3)
# print(y)
z = torch.zeros([2,5,3])
# print(z)
# print(z[0],'\n',z[1])
z = x + y
# print(z)
x_tensor = torch.randn(2,3)
y_numpy = np.random.randn(2,3)
# print(x.mm(y.t()))
x_numpy = x_tensor.numpy()
y_tensor = torch.from_numpy(y_numpy)
# print(torch.cuda.is_available())
# x = x.cuda()
# y = y.cuda()
# print(x + y)
x.cpu()

from torch.autograd import Variable
x = Variable(torch.ones(2,2),requires_grad=True)
# print(x)
y = x + 2
# print(y.grad_fn)
z = y * y
# print(z.grad_fn)
t = torch.mean(z)
# print(t)
t.backward()
# print(x.grad)
# print(y.grad)
# print(z.grad)

s = Variable(torch.FloatTensor([[0.01,0.02]]),requires_grad=True)
x = Variable(torch.ones(2,2),requires_grad=True)
for i in range(10):
    s = s.mm(x)
z = torch.mean(s)
z.backward()
# print(x.grad)
x =Variable(torch.linspace(0,100).type(torch.FloatTensor))
rand = Variable(torch.randn(100)) * 10
y = x + rand
x_train = x[:-10]
x_test = x[-10:]
y_train = y[:-10]
y_test = y[-10:]
import matplotlib.pyplot as plt


a = Variable(torch.rand(1),requires_grad=True)
b = Variable(torch.rand(1),requires_grad=True)
learning_rate = 0.0001
for i in range(1000):
    predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train)
    loss = torch.mean((predictions - y_train) ** 2)
    print('loss: ',loss)
    loss.backward()
    a.data.add_(-learning_rate * a.grad.data)
    b.data.add_(-learning_rate * b.grad.data)
    a.grad.data.zero_()
    b.grad.data.zero_()
plt.figure(figsize=(10,10))
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train.data.numpy(),predictions.data.numpy())
str1 = str(a.data.numpy()[0]) + 'x + ' + str(b.data.numpy()[0])
# plt.legend([xplot,yplot])
plt.show()

x_data = x_train.data.numpy()
x_pred = x_test.data.numpy()
plt.figure(figsize=(10,7))
plt.plot(x_data, y_train.data.numpy(),'o')
plt.plot(x_pred, y_test.data.numpy(),'s')
x_data = np.r_[x_data, x_test.data.numpy()]
plt.plot(x_data, a.data.numpy() * x_data + b.data.numpy())
plt.plot(x_pred, a.data.numpy() * x_pred + b.data.numpy(),'o')
plt.show()