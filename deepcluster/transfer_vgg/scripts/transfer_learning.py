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
from torchvision.models.resnet import resnet18

import ast
import argparse
import numpy as np
import time
import random
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
from data import make_data


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data_path', metavar='DIR', help='path to dataset')
    parser.add_argument('parameter_path',  help='path to parameter')

    parser.add_argument('--num_epoch', type=int, default=300,
                        help='number of total epochs to run (default: 300)')

    parser.add_argument('--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
  
    return parser.parse_args()

# モデルを学習させる関数を作成


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, file_name, f):
    loss_l_list_val = []
    loss_c_list_val = []
    loss_c_nega_list_val = []
    loss_c_posi_list_val = []
    
    loss_l_list_train = []
    loss_c_list_train = []
    loss_c_nega_list_train = []
    loss_c_posi_list_train = [] 
    
    softmax = nn.Softmax(dim=-1)
    tempo_val_loss = 100000000000000
    # GPUが使えるかを確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True
    early_stopping = EarlyStopping(patience=10, verbose=True, path='earlystopping.pth')
    # イテレーションカウンタをセット
    logs = []

    # epochのループ
    for epoch in range(num_epochs):
        iteration = 0
        val_iter = 0
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和
 
        
        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        
        f.write('-------------\n')
        f.write('Epoch {}/{}\n'.format(epoch+1, num_epochs))
        f.write('-------------\n')


        train_bbbb = []
        train_seikai = []
        val_bbbb = []
        val_seikai = []
        loss_ll_val = 0
        loss_cc_val = 0
        loss_c_posii_val = 0
        loss_c_negaa_val = 0
        loss_ll_train = 0
        loss_cc_train = 0
        loss_c_posii_train = 0
        loss_c_negaa_train = 0
        
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                print('（train）')
            else:
                net.eval()
    
            for images, targets in dataloaders_dict[phase]:
                if phase=='train':
                    images = torch.from_numpy(np.random.uniform(0.5, 1.8, size=(images.shape[0],1,1,1))) * images

                images = images.to(device, dtype=torch.float)
                targets = [ann.to(device, dtype=torch.float) for ann in targets]  # リストの各要素のテンソルをGPUへ
                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    # 順伝搬（forward）計算
                    outputs, decoded_box = net(images)

                    conf = softmax(outputs[1])
                    bbb = np.concatenate([conf[:, :, 1].to('cpu').detach().numpy()[:,:,None], 
                                             decoded_box.detach().numpy()], axis=2)


                    # 損失の計算
                    loss_l, loss_c, loss_c_posi, loss_c_nega = criterion(outputs, targets)
                    loss = loss_l + loss_c
#                     print('loss', loss)

            
#                     訓練時はバックプロパゲーション
                    if phase == 'train':
                        train_seikai.extend([ann.to('cpu').detach().numpy() for ann in targets])
                        train_bbbb.append(bbb)
                        loss_ll_train += loss_l.to('cpu').item()
                        loss_cc_train += loss_c.to('cpu').item()
                        loss_c_posii_train += loss_c_posi.to('cpu').item()
                        loss_c_negaa_train += loss_c_nega.to('cpu').item()
                        
                        loss.backward()  # 勾配の計算
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
                        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める

                        optimizer.step()  # パラメータ更新
                        epoch_train_loss += loss.item()
                        print("\r"+str(iteration)+'/'+str(int(17000/args.batch_size))+'       ', end="")
                        iteration += 1

                    # 検証時
                    else:
                        val_seikai.extend([ann.to('cpu').detach().numpy() for ann in targets])
                        val_bbbb.append(bbb)
#                         print(loss)
                        loss_ll_val += loss_l.to('cpu').item()
                        loss_cc_val += loss_c.to('cpu').item()
                        loss_c_posii_val += loss_c_posi.to('cpu').item()
                        loss_c_negaa_val += loss_c_nega.to('cpu').item()
                        
                        epoch_val_loss += loss.to('cpu').item()
                        val_iter += 1  
                        
        avg_train_loss = epoch_train_loss / iteration
        avg_val_loss = epoch_val_loss / val_iter
    
        f.write('\nepoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}\n'.format(epoch+1,
                                                                                  avg_train_loss,
                                                                                  avg_val_loss))
        print('\nepoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f} '.format(epoch+1,
                                                                                  avg_train_loss,
                                                                                  avg_val_loss))
        
        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        f.write('timer:  {:.4f} sec.\n'.format(t_epoch_finish - t_epoch_start))
        f.write('avarage_loss_l:{:.4f} ||avarage_loss_c:{:.4f} ||avarage_loss_c_posi:{:.4f} \
||avarage_loss_c_nega:{:.4f}\n'.format(loss_ll_val/val_iter, 
                                                   loss_cc_val/val_iter,
                                                   loss_c_posii_val/val_iter,
                                                   loss_c_negaa_val/val_iter
                                                  ))

        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        print('avarage_loss_l:{:.4f} ||avarage_loss_c:{:.4f} ||avarage_loss_c_posi:{:.4f} \
||avarage_loss_c_nega:{:.4f}'.format(loss_ll_val/val_iter,
                                                   loss_cc_val/val_iter,
                                                   loss_c_posii_val/val_iter,
                                                   loss_c_negaa_val/val_iter
                                                  ))
        
        loss_c_list_val.append(loss_cc_val/val_iter)
        loss_l_list_val.append(loss_ll_val/val_iter)
        loss_c_posi_list_val.append(loss_c_posii_val/val_iter)
        loss_c_nega_list_val.append(loss_c_negaa_val/val_iter)
        
        loss_c_list_train.append(loss_cc_train/iteration)
        loss_l_list_train.append(loss_ll_train/iteration)
        loss_c_posi_list_train.append(loss_c_posii_train/iteration)
        loss_c_nega_list_train.append(loss_c_negaa_train/iteration)
        
        train_f1_score = calc_f1score(train_seikai, np.concatenate(train_bbbb, axis=0))
        val_f1_score = calc_f1score(val_seikai, np.concatenate(val_bbbb, axis=0))
        f.write('train_f1_score :{:.4f}\n'.format(train_f1_score))
        f.write('val_f1_score :{:.4f}\n'.format(val_f1_score))
        print('train_f1_score :{:.4f}\n'.format(train_f1_score))
        print('val_f1_score :{:.4f}'.format(val_f1_score))

        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch+1,
                     'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                    'avarage_loss_l':loss_ll_val/val_iter, 'avarage_loss_c':loss_cc_val/val_iter,
                    'avarage_loss_c_posi':loss_c_posii_val/val_iter, 'avarage_loss_c_nega':loss_c_negaa_val/val_iter,
                    'val_f1_score':val_f1_score, 'train_f1_score':train_f1_score}

        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")
        if epoch_val_loss < tempo_val_loss:
            val_bbbb_ = np.concatenate(val_bbbb, axis=0)
            val_seikai_ = val_seikai
        tempo_val_loss = epoch_val_loss
        
#         early_stopping(val_f1_score, net)
        early_stopping(epoch_val_loss, net)
    
        if early_stopping.early_stop:
            train_bbbb = np.concatenate(train_bbbb, axis=0)
            #f.write(train_bbbb.shape, val_bbbb_.shape)
            print(train_bbbb.shape, val_bbbb_.shape)

            f.write('Early_Stopping\n')
            print('Early_Stopping')
            break
            
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        # ネットワークを保存する
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), 'ssd300_' +
                       str(epoch+1) + '.pth')
            
    return train_bbbb, val_bbbb_, train_seikai, val_seikai_,\
loss_l_list_val, loss_c_list_val, loss_c_posi_list_val, loss_c_nega_list_val,\
loss_l_list_train, loss_c_list_train, loss_c_posi_list_train, loss_c_nega_list_train




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

    train_label = pd.read_csv(args.data_path+'/train_label.csv')
    train_data = np.load(args.data_path+'/train.npy')
    val_label = pd.read_csv(args.data_path+'/val_label.csv')
    val_data = np.load(args.data_path+'/val.npy')

    train_data, val_data, train_label_list, val_label_list = make_data(train_data, val_data, train_label, val_label)

    train_sampler = NegativeSampler(train_data, true_size=12729, sample_negative_size=3591)
    val_sampler = NegativeSampler(val_data, true_size=252, sample_negative_size=292)
    # batch_size = 32

    train_dataset = DataSet(torch.Tensor(train_data), train_label_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=od_collate_fn)
    test_dataset = DataSet(torch.Tensor(val_data), val_label_list)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=od_collate_fn)


    dataloaders_dict = {"train": train_loader, "val": test_loader}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    

    
    net = SSD(phase='train', cfg=ssd_cfg)
    net = transfer_resnet(net, args.parameter_path)

    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=4, device=device)
    optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    net.to(device)

    f_log = open('log.txt', 'w')

    train_bbbb, val_bbbb, train_seikai, val_seikai,\
    loss_l_list_val, loss_c_list_val, loss_c_posi_list_val, loss_c_nega_list_val,\
    loss_l_list_train, loss_c_list_train, loss_c_posi_list_train, loss_c_nega_list_train\
    = train_model(net, dataloaders_dict , criterion, optimizer,
                            num_epochs=args.num_epoch, file_name='??',f=f_log)
    f_log.close()

    np.save('train_bbbb.npy', train_bbbb)
    np.save('val_bbbb.npy', val_bbbb)
    f = open('train_seikai.txt', 'wb')
    pickle.dump(train_seikai, f)
    f = open('val_seikai.txt', 'wb')
    pickle.dump(val_seikai, f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_l_list_train, label='loss_l_train')
    ax.plot(loss_c_list_train, label='loss_c_train')
    ax.plot(loss_l_list_val, label='loss_l_val')
    ax.plot(loss_c_list_val, label='loss_c_val')
    plt.legend()

    fig.savefig('train_loss.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_c_nega_list_train, label='loss_c_nega_train')
    ax.plot(loss_c_posi_list_train, label='loss_c_posi_train')
    ax.plot(loss_c_nega_list_val, label='loss_c_nega_val')
    ax.plot(loss_c_posi_list_val, label='loss_c_posi_val')
    plt.legend()
    fig.savefig('val_loss.png')

# plt.ylim(0, 4)

# plt.ylim(0, 4)

if __name__ == '__main__':
    args = parse_args()
    main(args)






