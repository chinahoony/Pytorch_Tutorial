# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 07:42:39 2021

@author: china
"""
# 't-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot'
import torchvision
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l
import torch.utils.data as Data
import sys,time

def show_fashionMNIST(images, labels):
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images),figsize=(12,12))
    
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    
    plt.show()

def text_labels(label):
    texts = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [texts[i] for i in label]
    

if __name__ == '__main__':
    root_dir = 'D:/02_python/study/Datasets/FashionMNIST'
    batch_size = 256
    train_MNIST = torchvision.datasets.FashionMNIST(root=root_dir, train=True,transform=torchvision.transforms.ToTensor(),download=True)
    test_MNIST = torchvision.datasets.FashionMNIST(root=root_dir, train=False,transform=torchvision.transforms.ToTensor(),download=True)
    
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
        
    train_iter = Data.DataLoader(train_MNIST, batch_size=batch_size,num_workers=num_workers,shuffle=True)
    test_iter = Data.DataLoader(test_MNIST, batch_size=batch_size,num_workers=num_workers,shuffle=True)
    
    x = [train_MNIST[i][0] for i in range(10)]
    y = [test_MNIST[i][1] for i in range(10)]
    
    show_fashionMNIST(x, text_labels(y))
    
    start = time.time()
    
    for X,y in train_iter:
        continue
    
    print(f'{time.time()-start:.2f} sec')
    
