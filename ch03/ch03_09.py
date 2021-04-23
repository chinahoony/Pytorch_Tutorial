# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 07:07:25 2021

@author: china
"""

import torch
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

num_inputs, num_outputs, num_hiddens = 784, 10, 256
batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

W1 = torch.tensor(np.random.normal(0.0, 0.1,(num_inputs, num_hiddens)),dtype=torch.float32)
b1 = torch.zeros(num_hiddens, dtype=torch.float32)
W2 = torch.tensor(np.random.normal(0.0,0.1,(num_hiddens, num_outputs)), dtype=torch.float32)
b2 = torch.zeros(num_outputs, dtype=torch.float32)

params = [W1, b1, W2, b2]

for param in params:
    param.requires_grad_(requires_grad=True)
    
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

def net(X):
    X = X.view(-1, num_inputs)
    H = relu(torch.matmul(X,W1)+b1)
    return torch.matmul(H, W2)+b2

loss = torch.nn.CrossEntropyLoss()
num_epochs = 5
lr = 100.

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)