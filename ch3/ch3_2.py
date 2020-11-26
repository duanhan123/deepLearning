import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim  as optim
import numpy as np

dataPath = 'hour.csv'
rides = pd.read_csv(dataPath)
# rides.head()
dummy_fields = ['season','weathersit','mnth','hr','weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each,drop_first=False)
    rides = pd.concat([rides,dummies], axis=1)
field_to_drop = ['instant','dteday','season','weathersit','weekday','atemp','mnth','workingday','hr']
data = rides.drop(field_to_drop, axis=1)
# print(data.shape)
quant_features = ['cnt','temp','hum','windspeed']
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(),data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

test_data = data[-21*24:]
train_data = data[:-21*24]
# print(train_data.head())
target_fields = ['cnt','casual','registered']
features, targets = train_data.drop(target_fields,axis=1),train_data[target_fields]
test_features, test_target = test_data.drop(target_fields,axis=1), test_data[target_fields]
X = features.values
Y = targets.cnt.values
Y = Y.astype(float)
Y = np.reshape(Y,[len(Y),1])
losses = []

inputSize = features.shape[1]
hiddenSize = 10
outputSize = 1
batchSize = 128
neu = torch.nn.Sequential(torch.nn.Linear(inputSize, hiddenSize), torch.nn.Sigmoid(), torch.nn.Linear(hiddenSize,outputSize))
cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(neu.parameters(), lr=0.01)
for i in range(1000):
    batchLoss = []
    for start in range(0,len(X),batchSize):
        end = start + batchSize if start + batchSize < len(X) else len(X)
        xx = torch.tensor(X[start:end],dtype=torch.float, requires_grad=True)
        yy = torch.tensor(Y[start:end],dtype=torch.float,requires_grad=True)
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batchLoss.append(loss.data.numpy())
        if i % 100 == 0:
            losses.append(np.mean(batchLoss))
            print(i, np.mean(batchLoss))
plt.plot(np.arange(len(losses))*100,losses)
plt.xlabel('epoch')
plt.ylabel('MSE')


targets = test_target['cnt']
targets = targets.values.reshape([len(targets),1])
targets = targets.astype(float)
x = torch.tensor(test_features.values,dtype=torch.float)
y = torch.tensor(targets,dtype=torch.float)

predict = neu(x)
predict = predict.data.numpy()
fig, ax = plt.subplots(figsize=(10,7))
mean, std = scaled_features['cnt']
ax.plot(predict * std + mean, label='Prediction')
ax.plot(targets * std + mean, label='Data')
ax.legend()
ax.set_xlabel('Data-time')
ax.set_ylabel('Counts')
dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24],rotation=45)


