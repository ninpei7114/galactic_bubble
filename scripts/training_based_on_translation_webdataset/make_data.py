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


def make_data(spitzer_path, validation_data_path, name, train_cfg, f_log):
    """
    学習に使用するtrain dataを作成する。
    validationは、性能を測るために固定とする。
    """

    ## Trainデータの作成
    ## train_dataのshapeは、(Num, 300, 300, 3)
    ## float32
    train_data, train_label = make_ring(spitzer_path, name, train_cfg)

    ## Trainデータをpngファイルに変換＋保存
    for i in range(train_data.shape[0]):
        pil_image = Image.fromarray(np.uint8(train_data[i]*255))
        pil_image.save('/workspace/dataset/train/Ring_%s.jpg'%i)
    
    ## Train labelをjsonに変換＋保存
    for i, row in train_label.iterrows():
        
        ll = []
        if len(row['xmin'])>=1:
            for la in range(len(row['xmin'])):
                ll.append({"Confidence":str(1), 
                        "XMin": str(row['xmin'][la]), "XMax": str(row['xmax'][la]), 
                        "YMin": str(row['ymin'][la]), "YMax": str(row['ymax'][la])})
        else:
            pass

        with open('/workspace/dataset/train/Ring_%s.json'%i, 'w') as f:
            json.dump(ll, f, indent=4)


    ## TrainとValのRingの枚数を取得
    val_Ring_num = len(glob.glob('/workspace/dataset/val/Ring_*.json'))
    train_Ring_num = len(glob.glob('/workspace/dataset/train/Ring_*.json'))


    # logに記入
    print_and_log(f_log, '====================================')
    print_and_log(f_log, '(confirm nan in Train)')
    print_and_log(f_log, 'Ring_data : %s'%np.isnan(np.sum(train_data)))
    print_and_log(f_log, ' ')
    print_and_log(f_log, '(confirm nan in Val)')
    print_and_log(f_log, ' ')


    Train_Non_Ring_num = len(glob.glob('/workspace/dataset/train/NonRing_*.json'))
    Val_Non_Ring_num = len(glob.glob('/workspace/dataset/val/NonRing_*.json'))
    print_and_log(f_log, 'Train Ring num  : %s '%train_Ring_num)
    print_and_log(f_log, 'Val Ring num  : %s '%val_Ring_num)
    print_and_log(f_log, ' ')
    print_and_log(f_log, 'Train Non-Ring num  : %s '%Train_Non_Ring_num)
    print_and_log(f_log, 'Val Non-Ring num  : %s '%Val_Non_Ring_num)
    print_and_log(f_log, ' ')
    print_and_log(f_log, 'Total Train num  : %s '%(train_Ring_num + Train_Non_Ring_num))
    print_and_log(f_log, 'Total Val num  : %s '%(val_Ring_num + Val_Non_Ring_num))
   
    with tarfile.open('/workspace/dataset/bubble_dataset_train.tar', 'w:gz') as tar:
        tar.add('/workspace/dataset/train')

    with tarfile.open('/workspace/dataset/bubble_dataset_val.tar', 'w:gz') as tar:
        tar.add('/workspace/dataset/val')


    return train_Ring_num, val_Ring_num
