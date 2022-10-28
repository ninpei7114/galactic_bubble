# from make_NonRing  import make_nonring
from make_Ring_data import make_ring
from sub import print_and_log

import numpy as np
import pandas as pd
import ast
from numpy.random import default_rng


def make_data(spitzer_path, validation_data_path, name, train_cfg, f_log):
    """
    学習に使用するtrain dataを作成する。
    validationは、性能を測るために固定とする。
    """

    # Trainデータの作成
    train_data, train_label = make_ring(spitzer_path, name, train_cfg)
    # train_label.to_csv(name+'/train_label.csv')
    # Validationデータの読み込み
    val_data = np.load(validation_data_path+'/val_ring.npy')
    val_label = pd.read_csv(validation_data_path+'/val_ring_label.csv')
    val_label['xmin'] = [ast.literal_eval(d) for d in val_label['xmin']]
    val_label['xmax'] = [ast.literal_eval(d) for d in val_label['xmax']]
    val_label['ymin'] = [ast.literal_eval(d) for d in val_label['ymin']]
    val_label['ymax'] = [ast.literal_eval(d) for d in val_label['ymax']]
    
    train_Ring_num, val_Ring_num = train_data.shape[0], val_data.shape[0]

    # TrainデータのNon-Ring
    # シード値を決める必要がある
    no_Ring_train = np.load('/workspace/NonRing/no_ring_300_21000_train.npy')
    # no_Ring_train_moyamoya = np.load('/workspace/NonRing/no_ring_moyamoya_train.npy')
    no_Ring_val = np.load('/workspace/NonRing/no_ring_300_900_val.npy')
    # no_Ring_val_moyamoya = np.load('/workspace/NonRing/no_ring_moyamoya_val.npy')

    train_arange = np.arange(0, no_Ring_train.shape[0])
    val_arange = np.arange(0, no_Ring_val.shape[0])
    no_Ring_train_random = default_rng(123).choice(train_arange, int(train_data.shape[0])*4, replace=False)
    # no_Ring_train_moyamoya_random = default_rng(123).integers(0, no_Ring_train_moyamoya.shape[0], int(train_data.shape[0]))
    no_Ring_val_random = default_rng(123).choice(val_arange, int(val_data.shape[0])*2, replace=False)
    # no_Ring_val_moyamoya_random = default_rng(123).integers(0, no_Ring_val_moyamoya.shape[0], int(val_data.shape[0]/2))

    # Non-Ringと合わせる
    print_and_log(f_log, '====================================')
    print_and_log(f_log, '(confirm nan in Train)')
    print_and_log(f_log, 'Ring_data : %s'%np.isnan(np.sum(train_data)))
    print_and_log(f_log, 'no_Ring_train : %s'%np.isnan(np.sum(no_Ring_train)))
    # print_and_log(f_log, 'no_Ring_train_moyamoya : %s'%np.isnan(np.sum(no_Ring_train_moyamoya)))
    print_and_log(f_log, ' ')
    print_and_log(f_log, '(confirm nan in Val)')
    print_and_log(f_log, 'Ring_data : %s'%np.isnan(np.sum(val_data)))
    print_and_log(f_log, 'no_Ring_val : %s'%np.isnan(np.sum(no_Ring_val)))
    # print_and_log(f_log, 'no_Ring_val_moyamoya : %s'%np.isnan(np.sum(no_Ring_val_moyamoya)))
    print_and_log(f_log, ' ')

    # train_data = np.concatenate([train_data, no_Ring_train[no_Ring_train_random], 
    #                          no_Ring_train_moyamoya[no_Ring_train_moyamoya_random]])
    train_data = np.concatenate([train_data, no_Ring_train[no_Ring_train_random]])

    # val_data = np.concatenate([val_data, no_Ring_val[no_Ring_val_random], 
    #                          no_Ring_val_moyamoya[no_Ring_val_moyamoya_random]])
    val_data = np.concatenate([val_data, no_Ring_val[no_Ring_val_random]])

    # Non-Ringのlabelと合わせる
    train_label = pd.concat([train_label, 
                        pd.DataFrame([{'fits':[],'name':[],'xmin':[],'xmax':[],'ymin':[],'ymax':[],'id':[] } 
                        for i in range(int(train_Ring_num)*4)])
                        ])
    val_label = pd.concat([val_label, 
                        pd.DataFrame([{'fits':[],'name':[],'xmin':[],'xmax':[],'ymin':[],'ymax':[],'id':[] } 
                        for i in range(int(val_Ring_num)*2)])
                        ])


    print_and_log(f_log, 'Train Ring num  : %s '%train_Ring_num)
    print_and_log(f_log, 'Val Ring num  : %s '%val_Ring_num)
    print_and_log(f_log, ' ')
    print_and_log(f_log, 'Total Train Ring Shape  : %s '%str(train_data.shape))
    print_and_log(f_log, 'Total Val Ring Shape  : %s '%str(val_data.shape))
    print_and_log(f_log, ' ')
    print_and_log(f_log, 'Total Train label length  : %s '%len(train_label))
    print_and_log(f_log, 'Total Val label length  : %s '%len(val_label))


    ## train_main.pyにて、samplerの数を書いている。


    train_label = train_label.reset_index()
    val_label = val_label.reset_index()
    
    # Non-Ringのラベルを追加
    train_label_list = []
    for i in range(len(train_label)):
        lab = [[train_label['xmin'][i][k], train_label['ymin'][i][k], train_label['xmax'][i][k], train_label['ymax'][i][k], 0] 
                for k in range(len(train_label['xmin'][i]))]

        train_label_list.append(np.array(lab))

    val_label_list = []
    for i in range(len(val_label)):
        lab = [[val_label['xmin'][i][k], val_label['ymin'][i][k], val_label['xmax'][i][k], val_label['ymax'][i][k], 0] 
                for k in range(len(val_label['xmin'][i]))]

        val_label_list.append(np.array(lab))

    # Train, Valデータのshapeは、(#, 300, 300, 3)
    # カラーを2chに
    train_data = train_data[:,:,:,:2]
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 2)

    val_data = val_data[:,:,:,:2]
    val_data = np.swapaxes(val_data, 2, 3)
    val_data = np.swapaxes(val_data, 1, 2)

    return train_data, train_label_list, val_data, val_label_list, train_Ring_num, val_Ring_num

