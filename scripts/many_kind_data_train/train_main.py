import pandas as pd
from math import sqrt as sqrt
from itertools import product as product
import pickle

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data
import torch.optim as optim
# from torchvision.models.resnet import resnet18

import ast
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from utils.ssd_model import SSD
from utils.ssd_model import MultiBoxLoss
from utils.ssd_model import decode

from sub import EarlyStopping
from sub import weights_init
from sub import calc_collision
from sub import calc_f1score
from sub import transfer_resnet

from data import od_collate_fn
from data import DataSet
from data import NegativeSampler
from make_data import make_data

from make_figure import make_figure
from train_model import train_model


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    # parser.add_argument('train_data_path', metavar='DIR', help='path to train dataset')
    # parser.add_argument('val_data_path', metavar='DIR', help='path to validation dataset')
    # parser.add_argument('Train_Ring_num', type=int, help='Ring number of train')
    # parser.add_argument('Train_Sampler_num', type=int, help='Non Ring sampler number of train')
    # parser.add_argument('Val_Ring_num', type=int, help='Ring number of validation')
    # parser.add_argument('Val_Sampler_num', type=int, help='Non Ring sampler number of validation')
    # parser.add_argument('parameter_path',  help='path to parameter')
    parser.add_argument('spitzer_path', metavar='DIR', help='spitzer_path')

    parser.add_argument('--num_epoch', type=int, default=300,
                        help='number of total epochs to run (default: 300)')

    parser.add_argument('--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
  
    return parser.parse_args()

# モデルを学習させる関数を作成





def main(args):
    input_size = 300
    color_mean = (0, 0)
    voc_classes = ['ring']

    ssd_cfg = {
        'num_classes': 2,  # 背景クラスを含めた合計クラス数
        'input_size': input_size,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    #     'bbox_aspect_num': [4, 4, 4, 4, 4, 4],
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ   
    #     'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2], [2]],
    }

    pattern = [[True, False], [False, True], [True, True]]
    
    for p in pattern:
        name = []
        for n, pp in zip(['flip', 'rotate'], p):
            if pp:
                name.append(n)
        name = '_'.join(name)
        os.mkdir(name)

        f_log = open(name+'/log.txt', 'w')

        train_data, train_label, val_data, val_label, train_Ring_num, val_Ring_num = make_data(args.spitzer_path, name, pattern, f_log)


        # print('Ring_num : ', args.Train_Ring_num)
        # print('Train_Non_Ring_Sampler : ', args.Train_Sampler_num)
        f_log.write('Train Negative Sampler num  : %s .\n'%train_Ring_num)
        f_log.write('Val Negative Sampler num  : %s .\n'%val_Ring_num)
        f_log.write('====================================\n')

        train_sampler = NegativeSampler(train_data, true_size=train_Ring_num, 
                                        sample_negative_size=train_Ring_num)
        val_sampler = NegativeSampler(val_data, true_size=val_Ring_num, 
                                        sample_negative_size=val_Ring_num)
        # batch_size = 32

        train_dataset = DataSet(torch.Tensor(train_data), train_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                        sampler=train_sampler, collate_fn=od_collate_fn)
        test_dataset = DataSet(torch.Tensor(val_data), val_label)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                        sampler=val_sampler, collate_fn=od_collate_fn)


        dataloaders_dict = {"train": train_loader, "val": test_loader}

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    

        
        net = SSD(phase='train', cfg=ssd_cfg)
        # net = transfer_resnet(net, args.parameter_path)

        criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=2, device=device)
        optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
        net.to(device)

        

        train_bbbb, val_bbbb_, train_seikai, val_seikai_,\
        loss_l_list_val, loss_c_list_val, loss_l_list_train, loss_c_list_train,\
        train_f1_score, val_f1_score\
        = train_model(net, dataloaders_dict , criterion, optimizer,
                                num_epochs=args.num_epoch, f=f_log, name=name, args=args, train_Ring_num=train_Ring_num)
        f_log.close()

        
        # 最もval_lossが低かった時の、正解ラベルとモデルのpredict結果
        np.save(name+'/train_bbbb.npy', train_bbbb)
        np.save(name+'/val_bbbb.npy', val_bbbb_)
        f = open(name+'/train_seikai.txt', 'wb')
        pickle.dump(train_seikai, f)
        f = open(name+'/val_seikai.txt', 'wb')
        pickle.dump(val_seikai_, f)


        ## lossの推移を描画する
        make_figure(name, loss_l_list_train, loss_c_list_train, loss_l_list_val, loss_c_list_val,
                    train_f1_score, val_f1_score)



if __name__ == '__main__':
    args = parse_args()
    main(args)