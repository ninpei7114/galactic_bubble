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
import glob

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

"""
example command:
python /workspace/galactic_bubble/scripts/training_based_on_translation_NoRing_cluster/train_main.py /dataset/spitzer_data/
 --savedir_path /workspace/webdataset_weights/NonRing_clustering/change_train_Region_3_split/change_train_Region_seed_123/Change_region_0/ 
 --NonRing_ratio 4 --augmentation_ratio 4 -s -i 0 -n 3 -r 125 --NonRing_class_num 7 --NonRing_remove_class_list 1 2 5
"""


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of SSD')
    parser.add_argument('spitzer_path', metavar='DIR', help='spitzer_path', 
                        default='/dataset/spitzer_data/')
    parser.add_argument('--validation_data_path', metavar='DIR', help='validation data path',
                        default='/workspace/val')
    parser.add_argument('--savedir_path', metavar='DIR', 
                        default='/workspace/weights/', help='savedire path  (default: /workspace/weights/)')
    parser.add_argument('--num_epoch', type=int, default=300,
                        help='number of total epochs to run (default: 300)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--NonRing_ratio', default=4, type=int,
                        help='Ring / NonRing ratio (default: 4)')
    parser.add_argument('--augmentation_ratio', default=4, type=int,
                        help='1 Ring augmentation ratio (default: 4)')
    parser.add_argument('--True_iou', default=0.5, type=float,
                        help='True IoU in MultiBoxLoss &  calc F1 score (default: 0.5)')

    parser.add_argument('--region_suffle', '-s', action='store_true')
    parser.add_argument('--fits_index', '-i', type=int)#, required=True)
    parser.add_argument('--n_splits', '-n', type=int, default=8)
    parser.add_argument('--fits_random_state', '-r', type=int, default=123)
    parser.add_argument('--NonRing_class_num', type=int, default=9)
    parser.add_argument('--NonRing_remove_class_list', nargs='*', type=int)
    parser.add_argument('--NonRing_mini_batch', type=int, default=48)
  
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
    flip_list = [False] #[False, True]
    rotate_list = [False] #[False, True]
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
        print_and_log(f_log, '###################')
        print_and_log(f_log, ' args parameters')
        print_and_log(f_log, '###################')
        print_and_log(f_log, 'augmentation_ratio : %s '%args.augmentation_ratio)
        print_and_log(f_log, 'region shuffle : %s '%args.region_suffle)
        print_and_log(f_log, 'fits_index : %s '%args.fits_index)
        print_and_log(f_log, 'n_splits : %s '%args.n_splits)
        print_and_log(f_log, 'fits_random_state : %s '%args.fits_random_state)


        ## png形式のRing画像とjson形式のlabelを作成
        train_Ring_num, Non_Ring_class_num = make_data(
            name, train_cfg, f_log, args)
        

        batch_size_ring = 16
        batch_size_nonring = args.NonRing_mini_batch
        
        # ds_train = webdataset.WebDataset("/%s/dataset/bubble_dataset_train.tar"%args.savedir_path).shuffle(1000000).decode("pil").to_tuple("png", "json").map(preprocess)
        # ds_val = webdataset.WebDataset("/%s/dataset/bubble_dataset_val.tar"%args.savedir_path).decode("pil").to_tuple("png", "json").map(preprocess)
        ds_ring_train = webdataset.WebDataset("/%s/dataset/%s/bubble_dataset_train_ring.tar"%(args.savedir_path, ''.join(name_))).shuffle(100000000000).decode("pil").to_tuple("png", "json").map(preprocess)
        NonRing_tar = [glob.glob("/%s/dataset/%s/bubble_dataset_train_nonring_*.tar"%(args.savedir_path, ''.join(name_)))]
        NonRing_rsample_num  = np.clip(train_Ring_num/batch_size_ring * batch_size_nonring / np.array(Non_Ring_class_num), 0, 1)
        NonRing_web_list = [
            webdataset.WebDataset(Nonring_tar_path).rsample(nr_rsample).shuffle(100000000000).decode("pil").to_tuple("png", "json").map(preprocess)
            for Nonring_tar_path, nr_rsample in zip(NonRing_tar, NonRing_rsample_num)]
        ds_val = webdataset.WebDataset("/%s/dataset/%s/bubble_dataset_val.tar"%(args.savedir_path, ''.join(name_))).decode("pil").to_tuple("png", "json").map(preprocess)


        dl_ring_train = torch.utils.data.DataLoader(ds_ring_train, collate_fn=od_collate_fn, batch_size=batch_size_ring)
        NonRing_dl_l = [
            torch.utils.data.DataLoader(nr_w_l, collate_fn=od_collate_fn, batch_size=int(batch_size_nonring/len(NonRing_web_list)))
            for nr_w_l in NonRing_web_list]
        dl_val = torch.utils.data.DataLoader(ds_val, collate_fn=od_collate_fn, batch_size=32)

        print_and_log(f_log, ' ')
        print_and_log(f_log, '====================================')


        dataloaders_dict = {"train": dl_ring_train, "val": dl_val}

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    

        
        net = SSD(cfg=ssd_cfg)

        criterion = MultiBoxLoss(jaccard_thresh=args.True_iou, neg_pos=3, device=device)
        optimizer = optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
        net.to(device)

        

        # train_bbbb, val_bbbb_, train_seikai, val_seikai_,\
        loss_l_list_val, loss_c_list_val, loss_l_list_train, loss_c_list_train,\
        train_f1_score, val_f1_score\
        = train_model(net, dataloaders_dict, NonRing_dl_l, criterion, optimizer,
                                num_epochs=args.num_epoch, f=f_log, name=name, args=args, train_Ring_num=train_Ring_num)
        f_log.close()


        ## lossの推移を描画する
        make_figure(name, loss_l_list_train, loss_c_list_train, loss_l_list_val, loss_c_list_val,
                    train_f1_score, val_f1_score)



if __name__ == '__main__':
    args = parse_args()
    main(args)