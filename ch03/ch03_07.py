# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 08:59:47 2021

@author: china
"""
import torch
from torch import nn
from collections import OrderedDict
from torch.nn import init
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    
    def forward(self, x):
        y = x.view(x.shape[0],-1)
        return y


num_inputs = 784
num_outputs = 10
batch_size = 256
num_epochs = 5

net = nn.Sequential(
        OrderedDict([
            ('flatten',FlattenLayer()),
            ('linear', nn.Linear(num_inputs, num_outputs))
            ])
        )
        
optimize = torch.optim.SGD(net.parameters(), lr=0.1)
init.normal_(net.linear.weight, 0.0, 0.01)
init.constant_(net.linear.bias, 0.0)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()


d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,None, None, optimize)
    
    
    
    