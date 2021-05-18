#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2021/5/18 11:02 AM
# Author  : xiaohui.wang
# File    : sim_classfiler_2.py
#使用sigmod，， bceloss


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x_layer):
        x_layer = torch.relu(self.n_hidden(x_layer))
        x_layer = self.out(x_layer)
        x_layer = torch.sigmoid(x_layer)
        return x_layer

net = Net(n_feature=2, n_hidden=10, n_output=1)
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.BCELoss()

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1)).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.FloatTensor)

for i in range(100):
    out = net(x)
    loss = loss_func(out.reshape(200), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# train result
train_result = net(x)
# print(train_result.shape)
print(train_result)
#train_predict = torch.max(train_result, 1)[1]
train_predict = train_result.reshape(200) >= 0.5
print(train_predict)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=train_predict.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


# test
t_data = torch.zeros(100, 2)
test_data = torch.normal(t_data, 5)
test_result = net(test_data)
print(test_result)
prediction = test_result.reshape(100) >= 0.5

plt.scatter(test_data[:, 0], test_data[:, 1], s=100, c=prediction.data.numpy(), lw=0, cmap='RdYlGn')
plt.show()