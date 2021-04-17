# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:12:07 2021

@author: china
"""

from IPython import display
import matplotlib.pyplot as plt
import random
import torch

def use_svg_display():
    display.set_matplotlib_formats('svg')
    
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def data_iter(batch_size, features, labels):
    n_data = len(features)
    indicies = list(range(n_data))
    random.shuffle(indicies)
    
    for i in range(0, n_data, batch_size):
        j = torch.LongTensor(indicies[i:min(i+batch_size, n_data)])
        yield features.index_select(0,j), labels.index_select(0,j)
        
def squard_loss(y_hat, y):
    return(y_hat-y.view(y_hat.size()))**2/2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= param.grad/batch_size * lr
        
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12,12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

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
                sgd(params, lr, batch_size)
            else:
                optimize.step()
            
            train_l_sum = l.item()
            train_acc_sum = (y_hat.argmax(dim=1)==y).float().sum().item()
            n = y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print(f'epoch {epoch+1}, loss {train_l_sum/n:.4f}, acc {train_acc_sum/n:.3f}, test acc {test_acc:.3f}')
    
    
    
    return