# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:16:24 2021

@author: china
"""

import torch
from torch.nn import init
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

num_inputs,num_hiddens,num_outputs = 784,256,10

net = torch.nn.Sequential(
    d2l.FlattenLayer(),
    torch.nn.Linear(num_inputs, num_hiddens),
    torch.nn.ReLU(),
    torch.nn.Linear(num_hiddens, num_outputs)
    )

for params in net.parameters():
    init.normal_(params, mean=0.0, std=0.01)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss = torch.nn.CrossEntropyLoss()

num_epochs = 5
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,None,None,optimizer)