#%%
import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#%%
image_size = 28
num_classes = 10
num_epochs = 20
batch_size = 64

#%%
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=False)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

#%%
indices = range(len(test_dataset))
indices_val = indices[:5000]
indices_test = indices[5000:]
# print(len(test_dataset),len(indices_test))
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

#%%
validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_val)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_test)

#%%

idx = 100
muteimg = train_dataset[idx][0].numpy()
# print(muteimg)
# print(muteimg.shape)
# print(muteimg[0,...].shape)
plt.imshow(muteimg[0,...])
plt.show()
print('标签是： ', train_dataset[idx][1])

#%%
depth = [4, 8]
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,4,5,padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
        self.fc2 = nn.Linear(512,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1,image_size // 4 * image_size // 4 * depth[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

    def retrieve_features(self,x):
        feature_map1 = F.relu(self.conv1(x))
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1, feature_map2)

#%%
def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
record = []
weights = []
for epoch in range(num_epochs):
    train_rights = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.clone().requires_grad_(True), target.clone().detach()
        net.train()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)

        if batch_idx % 100 == 0:
            net.eval()
            val_rights = []
            for (data, target) in validation_loader:
                data, target = data.clone().requires_grad_(True), target.clone().detach()
                output = net(data)
                right = rightness(output, target)
                val_rights.append(right)

            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]),sum([tup[1] for tup in val_rights]))
            print(val_r)
            print("训练周期： {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练准确率: {:.2f}%\t校验正确率: {:.2f}%".format(epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data, 100. * train_r[0].numpy() / train_r[1], 100. * val_r[0].numpy() / val_r[1]))
            record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))
            weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(), net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

#%%
plt.figure(figsize=(10,7))
plt.plot(record)
plt.xlabel('Steps')
plt.ylabel('Error rate')
plt.show()

#%%
net.eval()
vals = []
for data, target in test_loader:
    data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
    output = net(data)
    val = rightness(output, target)
    vals.append(val)

rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 100. * rights[0].numpy() / rights[1]
print(right_rate)

idx = 40
muteimg = test_dataset[idx][0].numpy()
plt.imshow(muteimg[0,...])
print('标签是: ', test_dataset[idx][1])
plt.show()

#%%
plt.figure(figsize=(10,7))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(net.conv1.weight.data.numpy()[i,0,...])
plt.show()

#%%
input_x = test_dataset[idx][0].unsqueeze(0)
feature_maps = net.retrieve_features(input_x)
plt.figure(figsize=(10,7))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(feature_maps[0][0,i,...].data.numpy())
plt.show()

#%%
plt.figure(figsize=(15,10))
for i in range(4):
    for j in range(8):
        plt.subplot(4,8,i*8+j+1)
        plt.axis('off')
        plt.imshow(net.conv2.weight.data.numpy()[j,i,...])
plt.show()

plt.figure(figsize=(10,7))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.axis('off')
    plt.imshow(feature_maps[1][0,i,...].data.numpy())
plt.show()
