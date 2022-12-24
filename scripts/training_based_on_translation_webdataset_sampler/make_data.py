# from make_NonRing  import make_nonring
from make_Ring_data import make_ring
from sub import print_and_log

import numpy as np
import pandas as pd
from numpy.random import default_rng
from PIL import Image

import ast
import json
import glob
import tarfile
import os
import shutil


def make_data(spitzer_path, validation_data_path, name, train_cfg, f_log, savedir_path, augmentation_ratio, NonRing_ratio):
    """
    学習に使用するtrain dataを作成する。
    validationは、性能を測るために固定とする。

    NonRing, Validationデータは毎回作成は高コストなため、
    事前に作成して、 cpする。

    """
    NonRing_rg = default_rng(123)
    ## Trainデータの作成
    ## train_dataのshapeは、(Num, 300, 300, 3)
    ## float32
    train_data, train_label = make_ring(spitzer_path, name, train_cfg, augmentation_ratio)

    save_data_path = savedir_path+''.join('dataset')+'/'+name.split('/')[-1]
    if os.path.exists(save_data_path):
        pass
    else:
        os.mkdir(save_data_path)
        os.mkdir(save_data_path+'/train')
        os.mkdir(save_data_path+'/train/ring')
        os.mkdir(save_data_path+'/train/nonring')
        # os.mkdir(save_data_path+'/val')


    ## Trainデータをpngファイルに変換＋保存
    for i in range(train_data.shape[0]):
        pil_image = Image.fromarray(np.uint8(train_data[i]*255))
        pil_image.save('%s/train/ring/Ring_%s.png'%(save_data_path, i))
    
    ## Train labelをjsonに変換＋保存
    for i, row in train_label.iterrows():
        
        ll = []
        if len(row['xmin'])>=1:
            for la in range(len(row['xmin'])):
                ll.append({"Confidence":str(0), 
                        "XMin": str(row['xmin'][la]), "XMax": str(row['xmax'][la]), 
                        "YMin": str(row['ymin'][la]), "YMax": str(row['ymax'][la])})
        else:
            pass

        with open('%s/train/ring/Ring_%s.json'%(save_data_path, i), 'w') as f:
            json.dump(ll, f, indent=4)
    
    ## NonRingのpngをcopyする。
    NonRing_origin = glob.glob('/workspace/NonRing_png/train/*.png')
    Choice_NonRing = NonRing_rg.choice(NonRing_origin, int(train_data.shape[0])*NonRing_ratio, replace=False)
    for i in Choice_NonRing:
        shutil.copyfile(i, '%s/train/nonring/%s'%(save_data_path, i.split('/')[-1]))
        shutil.copyfile(i[:-3]+'json', '%s/train/nonring/%s'%(save_data_path, i.split('/')[-1][:-3]+'json'))
    
    ## Validationデータをcopyする。
    shutil.copytree("/workspace/val", "%s/val"%save_data_path, dirs_exist_ok= True)

    ## TrainとValのRingの枚数を取得
    val_Ring_num = len(glob.glob('%s/val/Ring_*.json'%save_data_path))
    train_Ring_num = len(glob.glob('%s/train/ring/Ring_*.json'%save_data_path))


    # logに記入
    print_and_log(f_log, '====================================')
    print_and_log(f_log, 'Ring NonRing ratio = 1 : %s'%NonRing_ratio)
    print_and_log(f_log, ' ')
    print_and_log(f_log, '(confirm nan in Train)')
    print_and_log(f_log, 'Ring_data : %s'%np.isnan(np.sum(train_data)))
    print_and_log(f_log, ' ')
    print_and_log(f_log, '(confirm nan in Val)')
    print_and_log(f_log, ' ')


    Train_Non_Ring_num = len(glob.glob('%s/train/nonring/NonRing_*.json'%save_data_path))
    Val_Non_Ring_num = len(glob.glob('%s/val/NonRing_*.json'%save_data_path))
    print_and_log(f_log, 'Train Ring num  : %s '%train_Ring_num)
    print_and_log(f_log, 'Val Ring num  : %s '%val_Ring_num)
    print_and_log(f_log, ' ')
    print_and_log(f_log, 'Train Non-Ring num  : %s '%Train_Non_Ring_num)
    print_and_log(f_log, 'Val Non-Ring num  : %s '%Val_Non_Ring_num)
    print_and_log(f_log, ' ')
    print_and_log(f_log, 'Total Train num  : %s '%(train_Ring_num + Train_Non_Ring_num))
    print_and_log(f_log, 'Total Val num  : %s '%(val_Ring_num + Val_Non_Ring_num))
   
    with tarfile.open('%s/bubble_dataset_train_ring.tar'%save_data_path, 'w:gz') as tar:
        tar.add('%s/train/ring'%save_data_path)
    
    with tarfile.open('%s/bubble_dataset_train_nonring.tar'%save_data_path, 'w:gz') as tar:
        tar.add('%s/train/nonring'%save_data_path)

    with tarfile.open('%s/bubble_dataset_val.tar'%save_data_path, 'w:gz') as tar:
        tar.add('%s/val'%save_data_path)


    return train_Ring_num, val_Ring_num
