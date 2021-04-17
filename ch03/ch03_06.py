# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 07:42:39 2021

@author: china
"""

import torch
import torchvision
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp/partition

def net(X):
    return softmax(torch.mm(X.view(-1, num_inputs),W)+b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1,y.view(-1,1)))

def load_data_fashion_mnist(batch_size):
    train_dataset = torchvision.datasets.FashionMNIST(root = './Datasets/FashionMNIST', train=True, transform=torchvision.transforms.ToTensor(),download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    
    train_iter = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_iter, test_iter

def evaluate_accuracy(data_iter, net):
    sum_acc = 0.0
    n = 0
    for X, y in data_iter:
        sum_acc += (net(X).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return sum_acc / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params = None, lr=None, optimize=None):
    
    for epoch in range(num_epochs):
        train_l_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            if optimize is not None:
                optimize.grad_zero()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            
            if optimize is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimize.step()
            
            train_l_sum = l.item()
            train_acc_sum = (y_hat.argmax(dim=1)==y).float().sum().item()
            n = y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print(f'epoch {epoch+1}, loss {train_l_sum/n:.4f}, acc {train_acc_sum/n:.3f}, test acc {test_acc:.3f}')
    
    
    
    return


epoch_nums = 5
lr = 0.1
batch_size = 256

num_inputs = 784
num_outputs = 10


W = torch.tensor(np.random.normal(0.0,0.01,(num_inputs,num_outputs)), dtype = torch.float32)
b = torch.zeros(num_outputs, dtype=torch.float32)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

train_iter, test_iter = load_data_fashion_mnist(batch_size)

train_ch3(net, train_iter, test_iter, cross_entropy, epoch_nums, batch_size, params=[W, b], lr=lr)


X, y =iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])

