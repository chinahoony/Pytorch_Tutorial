# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:35:06 2021

@author: china
"""

from torch import nn
import torch
import numpy as np
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optimizer


def random_data_set(data_cnt, feature_cnt):
    w = [-1.2, 2.4]
    b = 4.2
    features = torch.randn(data_cnt, feature_cnt, dtype = torch.float32)
    labels = features[:,0]*w[0] + features[:,1]*w[1]+b
    labels += torch.tensor(np.random.normal(0.0,0.01,size=labels.size()),dtype=torch.float32)
    
    return features, labels


class LinearNet(nn.Module):
    def __init__(self, feature_cnt):
        super(LinearNet,self).__init__()
        self.mylinear = nn.Linear(feature_cnt, 1)
        
    def forward(self, x):
        y = self.mylinear(x)
        return y


if __name__ == '__main__':
    data_cnt = 1000
    feature_cnt = 2
    lr = 0.03
    batch_size = 10
    epoch_num=30
    
    features, labels = random_data_set(data_cnt, feature_cnt)
    
    data_set = Data.dataset.TensorDataset(features, labels)
    data_iter = Data.DataLoader(data_set,batch_size, shuffle=True)
    
    net = LinearNet(feature_cnt)
    
    init.normal_(net.mylinear.weight, 0.0, 0.01)
    init.constant_(net.mylinear.bias, 0.0)
    
    optim = optimizer.SGD(net.parameters(),lr=lr)
    
    lossfun = nn.MSELoss()
    
    for epoch in range(epoch_num):
        for x, y in data_iter:
            output = net(x)
            l = lossfun(output,y.view(output.size()))
            net.zero_grad()
            l.backward()
            optim.step()
        print(f'epoch:{epoch+1}, loss: {l.item()}')
    
    
    




