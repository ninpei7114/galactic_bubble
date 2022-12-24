import pandas as pd
from math import sqrt as sqrt
from itertools import product as product
import pickle

import torch
import torch.optim as optim
import webdataset
# from torchvision.models.resnet import resnet18

import argparse
import numpy as np
import os

from utils.ssd_model import SSD
from utils.ssd_model import MultiBoxLoss


from data import od_collate_fn
from data import preprocess
from data import DataSet
from data import NegativeSampler
from make_data import make_data

from make_figure import make_figure
from train_model import train_model
from sub import print_and_log
import itertools


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of SSD')
    parser.add_argument('spitzer_path', metavar='DIR', help='spitzer_path')
    parser.add_argument('validation_data_path', metavar='DIR', help='validation data path')
    parser.add_argument('--savedir_path', metavar='DIR', 
                        default='/workspace/weights/', help='savedire path  (default: /workspace/weights/)')
    parser.add_argument('--num_epoch', type=int, default=300,
                        help='number of total epochs to run (default: 300)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--NonRing_ratio', default=3, type=int,
                        help='Ring / NonRing ratio (default: 3)')
    parser.add_argument('--augmentation_ratio', default=10, type=int,
                        help='1 Ring augmentation ratio (default: 10)')
  
    return parser.parse_args()



# SSDの学習
def main(args):
    input_size = 300

    ssd_cfg = {
        'num_classes': 2,  # 背景クラスを含めた合計クラス数
        'input_size': input_size,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ   
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2], [2]],
    }

    # 上下反転、回転、縮小、平行移動の4パターンの組み合わせでaugmentatio を作る。
    flip_list = [False, True]
    rotate_list = [False, True]
    scale_list = [False]
    translation_list = [True]

    if os.path.exists(args.savedir_path):
        pass
    else:
        os.mkdir(args.savedir_path)

    for flip, rotate, scale, translation in itertools.product(flip_list, rotate_list, scale_list, translation_list):#, translation_list):
        train_cfg = {
            "flip": flip,
            "rotate": rotate,
            "scale": scale,
            "translation":translation
        }

        name_ = []
        # print('flip : %s,  rotate : %s,  scale : %s, translation : %s'%(flip, rotate, scale, translation))
        [name_.append(k+'_'+str(v)+'__') for k, v in zip(list(train_cfg.keys()), list(train_cfg.values()))]
        name = args.savedir_path+''.join(name_)
        if os.path.exists(name):
            pass
        else:
            os.mkdir(name)

        f_log = open(name+'/log.txt', 'w')
        print_and_log(f_log, 'flip : %s,  rotate : %s,  scale : %s,  translation : %s'%(flip, rotate, scale, translation))

        ## pngのRing画像とjson形式のlabelを作成
        train_Ring_num, val_Ring_num = make_data(
            args.spitzer_path, args.validation_data_path, name, train_cfg, f_log, 
            args.savedir_path, args.augmentation_ratio, args.NonRing_ratio)
        

        batch_size_ring = 16
        batch_size_nonring = 32
        
        # ds_train = webdataset.WebDataset("/%s/dataset/bubble_dataset_train.tar"%args.savedir_path).shuffle(1000000).decode("pil").to_tuple("png", "json").map(preprocess)
        # ds_val = webdataset.WebDataset("/%s/dataset/bubble_dataset_val.tar"%args.savedir_path).decode("pil").to_tuple("png", "json").map(preprocess)
        ds_ring_train = webdataset.WebDataset("/%s/dataset/%s/bubble_dataset_train_ring.tar"%(args.savedir_path, ''.join(name_))).shuffle(100000000000).decode("pil").to_tuple("png", "json").map(preprocess)
        ds_noring_train = webdataset.WebDataset("/%s/dataset/%s/bubble_dataset_train_nonring.tar"%(args.savedir_path, ''.join(name_))).rsample(0.1).shuffle(100000000000).decode("pil").to_tuple("png", "json").map(preprocess)
        ds_val = webdataset.WebDataset("/%s/dataset/%s/bubble_dataset_val.tar"%(args.savedir_path, ''.join(name_))).decode("pil").to_tuple("png", "json").map(preprocess)


        dl_ring_train = torch.utils.data.DataLoader(ds_ring_train, collate_fn=od_collate_fn, batch_size=batch_size_ring)
        dl_noring_train = torch.utils.data.DataLoader(ds_noring_train, collate_fn=od_collate_fn, batch_size=batch_size_nonring)
        dl_val = torch.utils.data.DataLoader(ds_val, collate_fn=od_collate_fn, batch_size=32)

        print_and_log(f_log, ' ')
        print_and_log(f_log, '====================================')


        dataloaders_dict = {"train": dl_ring_train, "val": dl_val}

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    

        
        net = SSD(cfg=ssd_cfg)

        criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=2, device=device)
        optimizer = optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
        net.to(device)

        

        # train_bbbb, val_bbbb_, train_seikai, val_seikai_,\
        loss_l_list_val, loss_c_list_val, loss_l_list_train, loss_c_list_train,\
        train_f1_score, val_f1_score\
        = train_model(net, dataloaders_dict, dl_noring_train, criterion, optimizer,
                                num_epochs=args.num_epoch, f=f_log, name=name, args=args, train_Ring_num=train_Ring_num)
        f_log.close()

        
        # 最もval_lossが低かった時の、正解ラベルとモデルのpredict結果
        # np.save(name+'/train_bbbb.npy', train_bbbb)
        # np.save(name+'/val_bbbb.npy', val_bbbb_)
        # f = open(name+'/train_seikai.txt', 'wb')
        # pickle.dump(train_seikai, f)
        # f = open(name+'/val_seikai.txt', 'wb')
        # pickle.dump(val_seikai_, f)


        ## lossの推移を描画する
        make_figure(name, loss_l_list_train, loss_c_list_train, loss_l_list_val, loss_c_list_val,
                    train_f1_score, val_f1_score)



if __name__ == '__main__':
    args = parse_args()
    main(args)