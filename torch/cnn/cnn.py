#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2021/5/18 5:44 PM
# Author  : xiaohui.wang
# File    : cnn.py
import torchvision
import torch
import torch.nn as nn

train_data = torchvision.datasets.MNIST(root="./mnist",
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)


import torch.utils.data as Data
BATCH_SIZE = 10
LR = 0.001
EPOCH = 10

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)
print(test_data.data.shape)
test_x = torch.unsqueeze(test_data.data, dim=1)/255
print(test_x.shape)

test_y = test_data.targets

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU()
        )

        self.out = nn.Linear(in_features=32*7*7, out_features=10)

    def forward(self, x):

        #(10000, 28, 28)
        #插入一维（10000,1,28,28)
        #取一个batch(10,1,28,28)
        #卷积（10,16,14,14)     ceil( n+2p-f/s + 1)  n:原尺寸 p:padding f:kernal_size s:步长
        #池化（10,16,7,7)
        #卷积（10,32,7,7)
        #全连接（10,10)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        output = torch.nn.functional.softmax(output,dim=1)
        return output

cnn = CNN()

optimzier = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()
import time
time_start = time.time()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

        if step % 50 == 0:
            test_output = cnn(test_x)

            #都是取最大值，所以输出经不经过softmax意义不大
            pred_y = torch.max(test_output, 1)[1].data.numpy()

            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)