# from make_NonRing  import make_nonring
from make_Ring_data import make_ring
from sub import print_and_log

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import ShuffleSplit
from PIL import Image

import json
import glob
import tarfile
import os
import shutil


def make_data(name, train_cfg, f_log, args):
    """
    学習に使用するtrain dataを作成する。
    validationは、性能を測るために固定とする。

    NonRing, Validationデータは毎回作成は高コストなため、
    事前に作成して、 copyする。

    """
    Data_rg = default_rng(123)

    ## 'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため使用しない
    fits_name = [
        'spitzer_00300+0000_rgb','spitzer_00600+0000_rgb','spitzer_00900+0000_rgb','spitzer_01200+0000_rgb',
        'spitzer_01500+0000_rgb','spitzer_01800+0000_rgb','spitzer_02100+0000_rgb','spitzer_02400+0000_rgb',
        'spitzer_02700+0000_rgb','spitzer_03000+0000_rgb','spitzer_03300+0000_rgb','spitzer_03600+0000_rgb',
        'spitzer_03900+0000_rgb','spitzer_04200+0000_rgb','spitzer_04500+0000_rgb','spitzer_04800+0000_rgb',
        'spitzer_05100+0000_rgb','spitzer_05400+0000_rgb','spitzer_05700+0000_rgb','spitzer_06000+0000_rgb',
        'spitzer_29700+0000_rgb','spitzer_30000+0000_rgb','spitzer_30300+0000_rgb','spitzer_30600+0000_rgb',
        'spitzer_30900+0000_rgb','spitzer_31200+0000_rgb','spitzer_31500+0000_rgb','spitzer_31800+0000_rgb',
        'spitzer_32100+0000_rgb','spitzer_32400+0000_rgb','spitzer_32700+0000_rgb','spitzer_33000+0000_rgb',
        'spitzer_33300+0000_rgb','spitzer_33600+0000_rgb','spitzer_33900+0000_rgb','spitzer_34200+0000_rgb',
        'spitzer_34500+0000_rgb','spitzer_34800+0000_rgb','spitzer_35100+0000_rgb','spitzer_35400+0000_rgb',
        'spitzer_35700+0000_rgb']

    if args.region_suffle:
        ss = ShuffleSplit(n_splits=args.n_splits, random_state=args.fits_random_state)
        train_index, val_index = list(ss.split(list(range(len(fits_name)))))[args.fits_index]
        train_l = [fits_name[i] for i in sorted(train_index)]
        val_l = [fits_name[i] for i in sorted(val_index)]
        print_and_log(f_log, 'This training is shuffled Training region ')
        print_and_log(f_log, '#################')
        print_and_log(f_log, '  training_region')
        print_and_log(f_log, '#################')
        print_and_log(f_log, str(train_l))
        print_and_log(f_log, ' ')
        print_and_log(f_log, '#################')
        print_and_log(f_log, '   val_region')
        print_and_log(f_log, '#################')
        print_and_log(f_log, str(val_l))
        print_and_log(f_log, ' ')
    
    else:
        ## 'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため使用しない
        train_l = [
        'spitzer_02100+0000_rgb','spitzer_04200+0000_rgb','spitzer_33300+0000_rgb','spitzer_35400+0000_rgb',
        'spitzer_00300+0000_rgb','spitzer_02400+0000_rgb','spitzer_04500+0000_rgb','spitzer_31500+0000_rgb',
        'spitzer_33600+0000_rgb','spitzer_35700+0000_rgb','spitzer_00600+0000_rgb','spitzer_02700+0000_rgb',
        'spitzer_04800+0000_rgb','spitzer_29700+0000_rgb','spitzer_31800+0000_rgb','spitzer_03000+0000_rgb',
        'spitzer_05100+0000_rgb','spitzer_30000+0000_rgb','spitzer_32100+0000_rgb','spitzer_01200+0000_rgb',
        'spitzer_03300+0000_rgb','spitzer_05400+0000_rgb','spitzer_30300+0000_rgb','spitzer_32400+0000_rgb',
        'spitzer_34500+0000_rgb','spitzer_01500+0000_rgb','spitzer_03600+0000_rgb','spitzer_05700+0000_rgb',
        'spitzer_30600+0000_rgb','spitzer_32700+0000_rgb','spitzer_34800+0000_rgb','spitzer_01800+0000_rgb',
        'spitzer_06000+0000_rgb','spitzer_30900+0000_rgb','spitzer_33000+0000_rgb','spitzer_35100+0000_rgb']
        train_l = sorted(train_l)

    #####################
    ## Trainデータの作成 ##
    #####################

    ## Trainデータの作成
    ## train_dataのshapeは、(Num, 300, 300, 3)
    ## float32
    train_data, train_label = make_ring(name, train_cfg, args, train_l)

    ## 必要なフォルダの作成
    save_data_path = args.savedir_path+''.join('dataset')+'/'+name.split('/')[-1]
    if os.path.exists(save_data_path):
        pass
    else:
        if os.path.exists(args.savedir_path+''.join('dataset')):
            pass 
        else:
            os.mkdir(args.savedir_path+''.join('dataset'))
        os.mkdir(save_data_path)
        os.mkdir(save_data_path+'/train')
        os.mkdir(save_data_path+'/train/ring')
        os.mkdir(save_data_path+'/train/nonring')
        for cl in range(args.NonRing_class_num):
            if cl in args.NonRing_remove_class_list:
                pass
            else:
                os.mkdir(save_data_path+'/train/nonring/class%s'%cl)

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


    #############################################
    ## Trainに用いるNon-Ringデータをコピー or 作成 ##
    #############################################
    Non_Ring_class_num = []
    if args.region_suffle:
        ## 領域ごとのNonRingをcopyする。
        
        for cl in range(args.NonRing_class_num):
            ## NonRingのクラスの内、使用しないクラスは除外する
            if cl in args.NonRing_remove_class_list:
                pass
            else:
                ## Non-RingのクラスごとにNonRingをコピーしていく
                NonRing_path = []
                _ = [glob.glob('/workspace/NonRing_png/region_NonRing_png/%s/class%s/*.png'%(i, cl)) for i in train_l]
                [NonRing_path.extend(i) for i in _]
                Non_Ring_class_num.append(len(NonRing_path))
                for i, k in enumerate(NonRing_path):
                    shutil.copyfile(k, '%s/train/nonring/class%s/%s'%(save_data_path, cl, 'NonRing_%s.png'%i))
                    shutil.copyfile(k[:-3]+'json', '%s/train/nonring/class%s/%s'%(save_data_path, cl, 'NonRing_%s.json'%i))
    else:
        ## デフォルトのNonRingをcopyする。
        NonRing_origin = glob.glob('/workspace/NonRing_png/default_NonRing_png/train/*.png')
        Choice_NonRing = Data_rg.choice(NonRing_origin, int(train_data.shape[0])*args.NonRing_ratio, replace=False)
        Non_Ring_class_num.append(len(NonRing_path))
        for i in Choice_NonRing:
            shutil.copyfile(i, '%s/train/nonring/%s'%(save_data_path, i.split('/')[-1]))
            shutil.copyfile(i[:-3]+'json', '%s/train/nonring/%s'%(save_data_path, i.split('/')[-1][:-3]+'json'))
    

    ###########################################
    ## Validationに用いるRing / NonRingをコピー ##
    ###########################################

    ## ********* 各領域ごとに *********
    if args.region_suffle:
        if os.path.exists('%s/val'%save_data_path):
            pass 
        else:
            os.mkdir('%s/val'%save_data_path)
        ## val_lで作成された、Ring / Non-Ringをコピーしてくる。
        ## Ringデータをコピーする。
        Val_origin = []
        a = [glob.glob('/workspace/val_png/region_val_png/%s/*.png'%i) for i in val_l]
        [Val_origin.extend(i) for i in a]
        for i, k in enumerate(Val_origin):
            shutil.copyfile(k, '%s/val/%s'%(save_data_path, 'Ring_%s.png'%i))
            shutil.copyfile(k[:-3]+'json', '%s/val/%s'%(save_data_path, 'Ring_%s.json'%i))
        
        ## Non-Ringをコピーする
        for cl in range(args.NonRing_class_num):
            ## NonRingのクラスの内、使用しないクラスは除外する
            if cl in args.NonRing_remove_class_list:
                pass
            else:
                NonRing_origin = []
                a = [glob.glob('/workspace/NonRing_png/region_NonRing_png/%s/class%s/*.png'%(i, cl)) for i in val_l]
                [NonRing_origin.extend(i) for i in a]
                Choice_NonRing = Data_rg.choice(NonRing_origin, int(len(Val_origin))*args.NonRing_ratio, replace=False)
                for i, k in enumerate(Choice_NonRing):
                    shutil.copyfile(k, '%s/val/%s'%(save_data_path, 'NonRing_%s.png'%i))
                    shutil.copyfile(k[:-3]+'json', '%s/val/%s'%(save_data_path, 'NonRing_%s.json'%i))


    ## ********* デフォルト領域で *********
    else:
        ## Validationデータをcopyする。
        ## Ringデータのコピー
        shutil.copytree("/workspace/val_png/default_val", "%s/val"%save_data_path, dirs_exist_ok= True)

        ## Non-Ringデータのコピー
        Val_default_path = glob.glob('/workspace/NonRing_png/default_NonRing_png/val/*.png')
        Choice_NonRing = Data_rg.choice(Val_default_path, 
                        int(len(glob.glob("%s/val/*"%save_data_path)))*args.NonRing_ratio, replace=False)
        for i, k in enumerate(Choice_NonRing):
            shutil.copyfile(k, '%s/val/%s'%(save_data_path, 'NonRing_%s.png'%i))
            shutil.copyfile(k[:-3]+'json', '%s/val/%s'%(save_data_path, 'NonRing_%s.json'%i))


    ## TrainとValのRingの枚数を取得
    val_Ring_num = len(glob.glob('%s/val/Ring_*.json'%save_data_path))
    train_Ring_num = len(glob.glob('%s/train/ring/Ring_*.json'%save_data_path))


    # logに記入
    print_and_log(f_log, '====================================')
    print_and_log(f_log, 'Ring NonRing ratio = 1 : %s'%args.NonRing_ratio)
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
    
    for cl in range(args.NonRing_class_num):
        ## NonRingのクラスの内、使用しないクラスは除外する
        if cl in args.NonRing_remove_class_list:
            pass
        else:
            with tarfile.open('%s/bubble_dataset_train_nonring_class%s.tar'%(save_data_path, cl), 'w:gz') as tar:
                tar.add('%s/train/nonring/class%s'%(save_data_path, cl))

    with tarfile.open('%s/bubble_dataset_val.tar'%save_data_path, 'w:gz') as tar:
        tar.add('%s/val'%save_data_path)


    return train_Ring_num, Non_Ring_class_num
