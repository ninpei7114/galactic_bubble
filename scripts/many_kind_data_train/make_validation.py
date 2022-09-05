import astropy.io.fits
import astropy.wcs

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import copy
import os
from torch.nn import functional as F
import proceesing
import label_caliculator
import ring_sub
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of SSD')
    parser.add_argument('spitzer_path', metavar='DIR', help='spitzer_path')
    parser.add_argument('validation_data_path', metavar='DIR', help='validation data path')
    parser.add_argument('--num_epoch', type=int, default=300,
                        help='number of total epochs to run (default: 300)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
  
    return parser.parse_args()


def main(args):

    val_l = ['spitzer_00900+0000_rgb','spitzer_03900+0000_rgb','spitzer_31200+0000_rgb','spitzer_34200+0000_rgb',
         'spitzer_33900+0000_rgb',]
    val_l = sorted(val_l)

    sig1 = 1/(2*(np.log(2))**(1/2))

    # choice catalogue from 'CH' or 'MWP'
    Ring_CATA = ring_sub.catalogue('MWP')

    # frame_mwp_train = []
    # mwp_ring_list_train = []
    frame_mwp_val = []
    mwp_ring_list_val = []

    l = val_l
    val_count = 0
    val_nan_count = 0
    def append_data(data, info, data_list, frame):
        if not np.isnan(data.sum()):
            data_list.append(data)
            frame.append(info)


    for i in range(len(l)): 

        fits_path = l[i]
        spitzer_rfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'r.fits')[0]
        spitzer_gfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'g.fits')[0]
        spitzer_bfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'b.fits')[0]

        #RGBにしたいため、fitsのdataを重ねる
        data = np.concatenate([proceesing.remove_nan(spitzer_rfits.data[:,:,None]), 
                                proceesing.remove_nan(spitzer_gfits.data[:,:,None]), 
                                proceesing.remove_nan(spitzer_bfits.data[:,:,None])], axis=2)


        a = data.shape[0]
        b = data.shape[1]
#         data[data!=data] = 0
        w = astropy.wcs.WCS(spitzer_rfits.header)
        GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
        GLON_max, GLAT_max = w.all_pix2world(0, a, 0) 

        GLON_center = (GLON_min+GLON_max)/2
        GLON_new_min = GLON_center-1.5
        GLON_new_max = GLON_center+1.5
        
        Ring_cata = Ring_CATA.query('@GLON_new_min < GLON <= @GLON_new_max')
        Ring_cata = Ring_cata.reset_index()
        train_count += len(Ring_cata)

        # star_listは辞書
        star_dic = label_caliculator.all_star(Ring_cata, w)
        print(fits_path)

        for _, row in Ring_cata.iterrows():
            flag, res_data, info = ring_sub.translation(Ring_CATA, fits_path, data, star_dic, 
            row, w, GLON_min, GLON_max, GLAT_min, GLAT_max)
            if flag:
                append_data(res_data, info, mwp_ring_list_val, frame_mwp_val)
        
    frame_mwp_val = pd.DataFrame(frame_mwp_val)
    frame_mwp_val['id']  = [i for i in range(len(frame_mwp_val))]

    mwp_ring_list_val = np.array(mwp_ring_list_val).astype(np.float32)
    val_Ring_num = mwp_ring_list_val.shape[0]

    no_Ring_val = np.load('NonRing/no_ring_300_900_val.npy')
    no_Ring_val_moyamoya = np.load('NonRing/no_ring_moyamoya_val.npy')
    no_Ring_val_random = np.random.randint(0, no_Ring_val.shape[0], int(mwp_ring_list_val.shape[0]/2))
    no_Ring_val_moyamoya_random = np.random.randint(0, no_Ring_val_moyamoya.shape[0], int(mwp_ring_list_val.shape[0]/2))


    mwp_ring_list_val = np.concatenate([mwp_ring_list_val, no_Ring_val[no_Ring_val_random], 
                             no_Ring_val_moyamoya[no_Ring_val_moyamoya_random]])

    for _ in range(int(val_Ring_num/2)*2):
        frame_mwp_val = pd.concat([frame_mwp_val, pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax', 'id'], 
             data= [[[] for i in range(7)]])])
    

    mwp_ring_list_val = mwp_ring_list_val[:,:,:,:2]
    mwp_ring_list_val = np.swapaxes(mwp_ring_list_val, 2, 3)
    mwp_ring_list_val = np.swapaxes(mwp_ring_list_val, 1, 2)

    np.save('val_data.npy', mwp_ring_list_val)
    frame_mwp_val.to_csv('val_label.csv')




if __name__ == '__main__':
    args = parse_args()
    main(args)




