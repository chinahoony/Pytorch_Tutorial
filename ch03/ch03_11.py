# -*- coding: utf-8 -*-
"""
Created on Sun May 16 21:50:20 2021

@author: china
"""

import torch
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l


n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train+n_test, 1))
poly_features = torch.cat((features, torch.pow(features,2), torch.pow(features,3)), 1)
labels = true_w[0]*poly_features[:,0] + true_w[1]*poly_features[:,1] + true_w[2]*poly_features[:,2]
labels += torch.tensor(np.random.normal(0.0, 0.01), dtype=torch.float32)
        
num_epochs=100
loss = torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1],1)
    
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1,1)
        test_labels = test_labels.view(-1,1)
        train_ls.append(loss(net(train_features),train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    
    print(f'final epoch: train loss {train_ls[-1]}, test loss {test_ls[-1]}')
    d2l.semilogy(range(1, num_epochs+1), train_ls, 'eopochs', 'loss', range(1, num_epochs+1), test_ls, ['train','test'])
    print('weight: ', net.weight.data, '\bias: ', net.bias.data)
    



# fit_and_plot(poly_features[:n_train,:], poly_features[n_train:,:], labels[:n_train], labels[n_train:])
# fit_and_plot(features[:n_train,:], features[n_train:,:], labels[:n_train], labels[n_train:])
fit_and_plot(poly_features[0:2,:], poly_features[n_train:,:], labels[0:2], labels[n_train:])

