# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from IPython import display
from matplotlib import pyplot as plt
import torch
import numpy as np
import random

def set_svg_display():
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize']=(3.5,2.4)
    
    
def random_data_set(data_cnt, feature_cnt):
    w = [2.4, -1.3]
    b = 4.2
    features = torch.randn(data_cnt, feature_cnt, dtype=torch.float32)
    labels = features[:,0]*w[0]+features[:,1]*w[1]+b
    labels += torch.tensor(np.random.normal(0.0, 0.02, size=labels.size()),dtype=torch.float32)
    
    return features, labels

def data_iter(features, labels, batch_size):
    data_cnt = len(features)
    indicies = list(range(data_cnt))
    random.shuffle(indicies)
    
    for i in range(0, data_cnt, batch_size):
        j = torch.LongTensor(indicies[i:min(i+batch_size,data_cnt)])
        yield features.index_select(0,j), labels.index_select(0, j)

def linreg(features, w, b):
    return torch.mm(features, w) + b

def lossfun(y, y_hat):
    return (y_hat-y.view(y_hat.size()))**2/2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size


if __name__ == '__main__':
    epoch_num = 3
    data_cnt = 1000
    feature_cnt = 2
    batch_size = 10
    lr = 0.02
    w = torch.tensor(np.random.normal(0.0, 0.01, size=(2, 1)),dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)
    
    features, labels = random_data_set(data_cnt, feature_cnt)
    
    set_svg_display()
    plt.scatter(features[:,0], labels, 2)
    
    net = linreg
    loss = lossfun
    
    w.requires_grad = True
    b.requires_grad = True
    
    for epoch in range(epoch_num):
        for x, y in data_iter(features, labels, batch_size):
            data = net(x, w, b)
            l = loss(data, y).sum()
            l.backward()
            sgd([w,b],lr, batch_size)
            
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print(f'epoch : {epoch+1} loss : {train_l.mean().item()}')
            
            
        