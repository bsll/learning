#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2021/5/19 3:42 PM
# Author  : xiaohui.wang
# File    : netLayer.py

import torch
import torch.nn as nn

# conv2d
def conv2d_s():
    input = torch.randn(20,16,50,100)

    m = nn.Conv2d(16, 33, 3, stride=2)

    output = m(input)

    print(output.shape)

    m = nn.Conv2d(16, 33, (3,5), stride=(2,1), padding=(4,2))
    output = m(input)

    #ceil((n + 2p - f) /s +1) * ceil((n + 2p - f) /s +1)
    #((50 + 2 * 4 - 3)/2 + 1）28
    #((100 + 2 * 2 - 5) + 1) 100
    print(output.shape)

#conv2d_s()
#torch.Size([20, 33, 24, 49])
#torch.Size([20, 33, 28, 100])

#poolLayer
def poolLayer():
    input = torch.randn(20, 16, 50, 32)
    m = nn.MaxPool2d(3, stride=2)

    #floor((n + 2p - f) /s +1) * floor((n + 2p - f) /s +1)
    #floor((50 - 3)/2 + 1) * floor((32-3)/2 + 1)
    output = m(input)
    print(output.shape)

    m = nn.MaxPool2d((3,2), stride=(2,1))
    output = m(input)
    print(output.shape)
#poolLayer()

#DropOut
def DropOut():
    input = torch.randn(10)
    m = nn.Dropout(p=0.8)
    output = m(input)
    print(output)
    input = torch.randn(10,5)
    m = nn.Dropout2d(p=0.8)
    output = m(input)
    print(output)

#DropOut()


#BN
def BN():
    input = torch.randn(20, 100, 35, 45)
    m = nn.BatchNorm2d(100)

    output = m(input)
    print(output.shape)

    m = nn.BatchNorm2d(100, affine=False)

    output = m(input)
    print(output.shape)

#BN()

#激活函数
#https://zhuanlan.zhihu.com/p/56210226
#torch.nn.LeakyReLU(), relu(), sigmod(), tanh(), softmax(), elu(), pRelu(),  RRelu(), Selu, SRelu
#APL  SoftPlus  bent identity


#损失函数
#https://zhuanlan.zhihu.com/p/97698386
#MSE MAE HuberLoss QuantileLoss BCE CrossEntropy   Hinge Loss
