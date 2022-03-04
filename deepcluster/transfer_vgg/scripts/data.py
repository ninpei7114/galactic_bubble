import pandas as pd
from math import sqrt as sqrt
from itertools import product as product

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
from torchsummary import summary
import torch.utils.data as data
import torch.optim as optim

import ast
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, dim=0)
        
    return imgs, targets

class DataSet():
    def __init__(self, data, label):
        self.label = label
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


class NegativeSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, source, true_size, sample_negative_size):
        self.true_size = true_size
        self.negative_size = len(source) - true_size
        self.sample_negative_size = sample_negative_size
    def __iter__(self):
        neg = np.arange(self.true_size, self.true_size + self.sample_negative_size)
        indeces = np.concatenate((np.arange(self.true_size), np.random.choice(neg, self.sample_negative_size)))
        np.random.shuffle(indeces)
        for i in indeces:
            yield i
    def __len__(self):
        return self.true_size + self.sample_negative_size


def make_data(train_data, val_data, train_label, val_label):
    train_data = train_data[:,:,:,:2]
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 2)

    val_data = val_data[:,:,:,:2]
    val_data = np.swapaxes(val_data, 2, 3)
    val_data = np.swapaxes(val_data, 1, 2)

    print('train label : ', len(train_label), 'train data : ', train_data.shape)
    print('val label : ', len(val_label), 'val data : ', val_data.shape)

    train_label = train_label.drop('Unnamed: 0', axis=1)
    val_label = val_label.drop('Unnamed: 0', axis=1)

    train_label['xmin'] = [ast.literal_eval(d) for d in train_label['xmin']]
    train_label['xmax'] = [ast.literal_eval(d) for d in train_label['xmax']]
    train_label['ymin'] = [ast.literal_eval(d) for d in train_label['ymin']]
    train_label['ymax'] = [ast.literal_eval(d) for d in train_label['ymax']]

    val_label['xmin'] = [ast.literal_eval(d) for d in val_label['xmin']]
    val_label['xmax'] = [ast.literal_eval(d) for d in val_label['xmax']]
    val_label['ymin'] = [ast.literal_eval(d) for d in val_label['ymin']]
    val_label['ymax'] = [ast.literal_eval(d) for d in val_label['ymax']]

    train_label = train_label.reset_index()
    val_label = val_label.reset_index()

    train_label_list = []
    for i in range(len(train_label)):
        lab = []
        for k in range(len(train_label['xmin'][i])):
            labe = []
            labe.append(train_label['xmin'][i][k])
            labe.append(train_label['ymin'][i][k])
            labe.append(train_label['xmax'][i][k])
            labe.append(train_label['ymax'][i][k])
            labe.append(0)
            lab.append(labe)
        train_label_list.append(np.array(lab))

    val_label_list = []
    for i in range(len(val_label)):
        lab = []
        for k in range(len(val_label['xmin'][i])):
            labe = []
            labe.append(val_label['xmin'][i][k])
            labe.append(val_label['ymin'][i][k])
            labe.append(val_label['xmax'][i][k])
            labe.append(val_label['ymax'][i][k])
            labe.append(0)
            lab.append(labe)
        val_label_list.append(np.array(lab))

    return train_data, val_data, train_label_list, val_label_list
