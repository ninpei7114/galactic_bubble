import pandas as pd
from math import sqrt as sqrt
from itertools import product as product
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
# from torchvision.models.resnet import resnet18

import numpy as np
from numpy.random import default_rng
import time

from utils.ssd_model import Detect
from sub import EarlyStopping
from sub import EarlyStopping_f1_score
from sub import weights_init
from sub import calc_f1score
from sub import print_and_log


def train_model(net, dataloaders_dict, NonRing_dl_l, criterion, optimizer, num_epochs, f, name, args, train_Ring_num):
    """
    net : モデル
    dataloaders_dict : {"train": dl_ring_train, "val": dl_val}
    NonRing_dl_l : TrainingのNon-RingのDataloaderのList
    
    num_epochs : epoch数
    f : ログを取るためのファイル
    name : 保存するためのディレクトリ
    train_Ring_num : TrainingのRingの数
    """

    loss_l_list_val = []
    loss_c_list_val = []
    loss_c_nega_list_val = []
    loss_c_posi_list_val = []
    
    loss_l_list_train = []
    loss_c_list_train = []
    loss_c_nega_list_train = []
    loss_c_posi_list_train = [] 
    
    # GPUが使えるかを確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print_and_log(f, "使用デバイス： {}".format(device))

    # ネットワークをGPUへ
    net.to(device)

    net.vgg.apply(weights_init)
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True
    # early_stopping = EarlyStopping(patience=10, verbose=True, path=name+'/earlystopping.pth', flog=f)
    early_stopping = EarlyStopping_f1_score(patience=10, verbose=True, path=name+'/earlystopping.pth', flog=f)
    # イテレーションカウンタをセット
    logs = []
    train_rng = default_rng(123)

    # epochのループ
    for epoch in range(num_epochs):
        iteration = 0
        val_iter = 0
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和
 
        # 開始時刻を保存
        t_epoch_start = time.time()

        print_and_log(f, '-------------')
        print_and_log(f, 'Epoch {}/{}'.format(epoch+1, num_epochs))
        print_and_log(f, '-------------')


        train_bbbb_loc = []
        train_bbbb_conf = []
        train_bbbb_b = []
        train_seikai = []
        val_bbbb_loc = []
        val_bbbb_conf = []
        val_bbbb_b = []
        val_seikai = []
        train_f1_score_l = []
        train_f1_score_l_non_ring = []
        val_f1_score_l = []
        val_f1_score_l_non_ring = []

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
                iter_noring_list = [dliter.__iter__() for dliter in NonRing_dl_l]
                # iter_noring = dl_noring_train.__iter__()
                print_and_log(f, ' (train) ')
            else:
                print('val')
                net.eval()
    
            for images, targets in dataloaders_dict[phase]:
                if phase=='train':
                    # images = torch.from_numpy(train_rng.uniform(0.5, 1.8, size=(images.shape[0],1,1,1))) * images
                    no_ring_list = [next(iter_noring, None) for iter_noring in iter_noring_list]
                    # noring = next(iter_noring, None)
                    # if noring is None:
                    if no_ring_list[0] is None:
                        # iter_noring = dl_noring_train.__iter__() # 最後まで行っていたら最初に戻して
                        iter_noring_list = [dliter.__iter__() for dliter in NonRing_dl_l]
                        no_ring_list = [next(iter_noring, None) for iter_noring in iter_noring_list]
                        # noring = next(iter_noring) 
                    no_ring_image = [noring[0] for noring in no_ring_list]
                    no_ring_target = [noring[1] for noring in no_ring_list]
                    images = np.concatenate((images, np.array(no_ring_image)))
                    targets = targets + no_ring_target
                    images = torch.from_numpy(images)
                else:
                    images = torch.from_numpy(images)

                images = images.permute(0, 3, 1, 2)[:,:2,:,:]
                images = images.to(device, dtype=torch.float)
                targets = [ann.to(device, dtype=torch.float) for ann in targets]  # リストの各要素のテンソルをGPUへ
                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    # 順伝搬（forward）計算
                    outputs, decoded_box = net(images)

                    # 損失の計算
                    loss_l, loss_c, loss_c_posi, loss_c_nega = criterion(outputs, targets)
                    loss = loss_l + loss_c
            
#                     訓練時はバックプロパゲーション
                    if phase == 'train':
                        train_seikai.extend([ann.to('cpu').detach().numpy() for ann in targets])
                        
                        loss_ll_train += loss_l.to('cpu').item()
                        loss_cc_train += loss_c.to('cpu').item()
                        loss_c_posii_train += loss_c_posi.to('cpu').item()
                        loss_c_negaa_train += loss_c_nega.to('cpu').item()
                        train_bbbb_loc.append(outputs[0].to('cpu'))
                        train_bbbb_conf.append(outputs[1].to('cpu'))
                        train_bbbb_b.append(outputs[2].to('cpu'))
                        
                        loss.backward()  # 勾配の計算
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
                        # nn.utils.clip_grad_norm_(net.parameters(), clip_value=10.0)
                        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配10.0に留める

                        optimizer.step()  # パラメータ更新
                        epoch_train_loss += loss.item()
                        print("\r"+str(iteration)+'/'+str(int((
                            int(train_Ring_num)+int(train_Ring_num*3))/args.batch_size))+'       ', end="")
                        iteration += 1

                    # 検証時
                    else:
                        val_seikai.extend([ann.to('cpu').detach().numpy() for ann in targets])
                        val_bbbb_loc.append(outputs[0].to('cpu'))
                        val_bbbb_conf.append(outputs[1].to('cpu'))
                        val_bbbb_b.append(outputs[2].to('cpu'))

                        loss_ll_val += loss_l.to('cpu').item()
                        loss_cc_val += loss_c.to('cpu').item()
                        loss_c_posii_val += loss_c_posi.to('cpu').item()
                        loss_c_negaa_val += loss_c_nega.to('cpu').item()
                        
                        epoch_val_loss += loss.to('cpu').item()
                        val_iter += 1  

        val_bbbb = [torch.cat(val_bbbb_loc), torch.cat(val_bbbb_conf), val_bbbb_b[0]] 
        train_bbbb = [torch.cat(train_bbbb_loc), torch.cat(train_bbbb_conf), train_bbbb_b[0]]      
        avg_train_loss = epoch_train_loss / iteration
        avg_val_loss = epoch_val_loss / val_iter
    
        print_and_log(f, '\nepoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f} '.format(epoch+1,
                                                                                  avg_train_loss,
                                                                                  avg_val_loss))
        
        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()

        print_and_log(f, 'time:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        print_and_log(f, 'avarage_loss_l:{:.4f} ||avarage_loss_c:{:.4f} ||avarage_loss_c_posi:{:.4f} \
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
        
        train_f1_score, train_threthre, train_f1_score_non_ring, train_threthre_noring = calc_f1score(train_seikai, train_bbbb, mode='train',iou=args.True_iou, top_k=100)
        val_f1_score, val_threthre, val_f1_score_non_ring, val_threthre_noring = calc_f1score(val_seikai, val_bbbb, mode='val', iou=args.True_iou)

        print_and_log(f, 
        'train_f1_score : {:.4f}, threshold : {:.4f}'.format(train_f1_score, train_threthre))
        print_and_log(f, 
        'train_f1_score_add_non_ring : {:.4f}, threshold : {:.4f}\n'.format(train_f1_score_non_ring, train_threthre_noring))
        print_and_log(f, 
        'val_f1_score : {:.4f}, threshold : {:.4f}'.format(val_f1_score, val_threthre))
        print_and_log(f, 
        'val_f1_score_add_non_ring : {:.4f}, threshold_add_non_ring : {:.4f}\n'.format(val_f1_score_non_ring, val_threthre_noring))
        train_f1_score_l.append(train_f1_score)
        train_f1_score_l_non_ring.append(train_f1_score_non_ring)
        val_f1_score_l.append(val_f1_score)
        val_f1_score_l_non_ring.append(val_f1_score_non_ring)

        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {
                    'epoch': epoch+1,
                    'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                    'avarage_loss_l':loss_ll_val/val_iter, 'avarage_loss_c':loss_cc_val/val_iter,
                    'avarage_loss_c_posi':loss_c_posii_val/val_iter, 'avarage_loss_c_nega':loss_c_negaa_val/val_iter,
                    'val_f1_score':val_f1_score, 'val_f1_score_non_ring':val_f1_score_non_ring, 
                    'train_f1_score':train_f1_score, 'train_f1_score_non_ring':train_f1_score_non_ring}

        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(name+'/log_output.csv')  
        
        early_stopping(val_f1_score, net)
        # early_stopping(epoch_val_loss, net)

        if early_stopping.counter == 0:
            # np.save('train_bbbb.npy', np.concatenate(train_bbbb, axis=0))
            f_early_ = open(name+'/train_bbbb.txt', 'wb')
            pickle.dump(train_bbbb, f_early_)
            # np.save('val_bbbb.npy', np.concatenate(val_bbbb, axis=0))
            f_early_ = open(name+'/val_bbbb.txt', 'wb')
            pickle.dump(val_bbbb, f_early_)
            f_early_ = open(name+'/train_seikai.txt', 'wb')
            pickle.dump(train_seikai, f_early_)
            f_early_ = open(name+'/val_seikai.txt', 'wb')
            pickle.dump(val_seikai, f_early_)
    
        if early_stopping.early_stop:
            print_and_log(f, 'Early_Stopping')
            break
            
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        # ネットワークを保存する
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), name+'/ssd300_' +
                       str(epoch+1) + '.pth')

    return loss_l_list_val, loss_c_list_val, loss_l_list_train, loss_c_list_train,\
train_f1_score_l, val_f1_score_l