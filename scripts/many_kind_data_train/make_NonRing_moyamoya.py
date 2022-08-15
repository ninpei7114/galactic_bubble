import astropy.io.fits
import astroquery.vizier
import astropy.wcs
from astropy.coordinates import SkyCoord

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import time
import pathlib
import random
import collections
import copy
import argparse

import torch
from torch.nn import functional as F
from torch import nn

import proceesing
import NonRing_sub



def parse_args():
    parser = argparse.ArgumentParser(description='make data for deepcluster')

    parser.add_argument('fits_path', metavar='DIR', help='path to dataset')
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')
    # parser.add_argument('mwp_catalogu_path', metavar='DIR', help='path to mwp catalogue')
    # # parser.add_argument('augmentation_num', metavar='DIR', help='the number of augmentation')
    # parser.add_argument('savedir_format', metavar='DIR', help='data save dir name format')
    # parser.add_argument('--augmentation_num_list', required=True, nargs="*", type=int, help='the number of augmentation')
    # parser.add_argument('--savedir', default='ring_to_circle_nan_fits', 
    #                     help='data save dir')

    return parser.parse_args()



def main(args):

    val_l = ['spitzer_00900+0000_rgb','spitzer_03900+0000_rgb','spitzer_31200+0000_rgb','spitzer_34200+0000_rgb','spitzer_33900+0000_rgb',]
    val_l = sorted(val_l)

    train_l = ['spitzer_02100+0000_rgb','spitzer_04200+0000_rgb','spitzer_33300+0000_rgb','spitzer_35400+0000_rgb','spitzer_00300+0000_rgb',
        'spitzer_02400+0000_rgb','spitzer_04500+0000_rgb','spitzer_31500+0000_rgb','spitzer_33600+0000_rgb','spitzer_35700+0000_rgb',
        'spitzer_00600+0000_rgb','spitzer_02700+0000_rgb','spitzer_04800+0000_rgb','spitzer_29700+0000_rgb','spitzer_31800+0000_rgb',
        'spitzer_03000+0000_rgb','spitzer_05100+0000_rgb','spitzer_30000+0000_rgb','spitzer_32100+0000_rgb','spitzer_01200+0000_rgb',
        'spitzer_03300+0000_rgb','spitzer_05400+0000_rgb','spitzer_30300+0000_rgb','spitzer_32400+0000_rgb','spitzer_34500+0000_rgb',
        'spitzer_01500+0000_rgb','spitzer_03600+0000_rgb','spitzer_05700+0000_rgb','spitzer_30600+0000_rgb','spitzer_32700+0000_rgb',
        'spitzer_34800+0000_rgb','spitzer_01800+0000_rgb','spitzer_06000+0000_rgb','spitzer_30900+0000_rgb','spitzer_33000+0000_rgb',
        'spitzer_35100+0000_rgb']
    train_l = sorted(train_l)
    #,'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため、x


    hist = np.array([190., 138.,  97.,  55.,  29.,  10.,  12.,   8.,  12.,  11.,   4.,
            7.,   4.,   5.,   1.,   1.,   2.,   0.,   1.,   1.,   0.,   0.,
            1.,   0.,   0.,   0.,   1.,   0.,   0.,   1.])
    hist_ = hist/591
    hisy = hist_.tolist()
    range_ = np.array([ 0.11      ,  0.72866666,  1.3473333 ,  1.966     ,  2.5846667 ,
            3.2033334 ,  3.822     ,  4.4406667 ,  5.0593333 ,  5.678     ,
            6.2966666 ,  6.9153333 ,  7.534     ,  8.152667  ,  8.771334  ,
            9.39      , 10.008667  , 10.627334  , 11.246     , 11.864667  ,
            12.483334  , 13.102     , 13.720667  , 14.339334  , 14.958     ,
            15.576667  , 16.195333  , 16.814     , 17.432667  , 18.051332  ,
            18.67      ])


    for mode in ['train', 'val']:

        if mode == 'train':
            epoch = 100
            ref_path_list = train_l
            choice_num = len(train_l)-1
        else:
            epoch = 80
            ref_path_list = val_l
            choice_num = len(val_l)-1


        no_nan_no_ring_list = []
        start = time.time()
        sig1 = 1/(2*(np.log(2))**(1/2))
        fits_path = pathlib.Path(args.fits_path)
        for i in range(epoch):
            random_int = random.randint(0, choice_num)
            
            path = ref_path_list[random_int]
            spitzer_rfits = astropy.io.fits.open(fits_path/path/'r.fits')[0]
            spitzer_gfits = astropy.io.fits.open(fits_path/path/'g.fits')[0]   
            spitzer_bfits = astropy.io.fits.open(fits_path/path/'b.fits')[0]   
            header = spitzer_rfits.header
            w = astropy.wcs.WCS(header)
            data = np.concatenate([spitzer_rfits.data[:,:,None], 
                                spitzer_gfits.data[:,:,None], 
                                spitzer_bfits.data[:,:,None]], axis=2)
            
            print(path)
        #     data[data != data] = 0
            # GLON_LAT関数でGLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1を出す
            GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1 = NonRing_sub.GLON_LAT(data, header, w)
            
            for k in range(100):

                cut_data = NonRing_sub.no_nan_ring(data, GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1, hist_, hisy, range_, w)
        #         print(cut_data.shape)
                pi = proceesing.conv(300, sig1, cut_data)
                r_shape_y = pi.shape[0]
                r_shape_x = pi.shape[1]
                res_data = pi[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
                res_data = proceesing.normalize(res_data)
                res_data = proceesing.resize(res_data, 300)
                
                if np.sum(res_data[:,:,:2]) >= 20000:
                    no_nan_no_ring_list.append(res_data)
                
                
        stop = time.time()
        print((stop-start)/60)

        np.save('NonRing/no_ring_moyamoya_%s.npy'%mode, np.array(no_nan_no_ring_list))



if __name__ == '__main__':
    args = parse_args()
    main(args)


