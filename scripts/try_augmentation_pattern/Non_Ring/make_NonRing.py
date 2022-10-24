import astropy.io.fits
import astropy.wcs

import numpy as np
from npy_append_array import NpyAppendArray
from numpy.random import default_rng

import time
import pathlib
import random
import argparse
from tqdm import tqdm

import proceesing
import NonRing_sub



def parse_args():
    parser = argparse.ArgumentParser(description='make data for deepcluster')

    parser.add_argument('fits_path', metavar='DIR', help='path to dataset')
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')

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
    random_uni = default_rng(123)

    for mode in ['train', 'val']:

        if mode == 'train':
            epoch = 400
            ref_path_list = train_l
            choice_num = len(train_l)-1
        else:
            epoch = 30
            ref_path_list = val_l
            choice_num = len(val_l)-1

        no_nan_no_ring_list = NpyAppendArray('/workspace/NonRing/no_ring_300_%s_%s.npy'%(epoch*30, mode))

        start = time.time()
        sig1 = 1/(2*(np.log(2))**(1/2))
        fits_path = pathlib.Path(args.fits_path)
        pbar = tqdm(range(epoch))
        for _ in pbar:
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
            
            pbar.set_description(path)

            NonRing_sub_c = NonRing_sub.NonRing_sub(w, data, random_uni)
        #     data[data != data] = 0
            # GLON_LAT関数でGLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1を出す
            NonRing_sub_c.GLON_LAT(data, header)
            
            for _ in range(30):

                cut_data = NonRing_sub_c.no_nan_ring()
        #         print(cut_data.shape)
                pi = proceesing.conv(300, sig1, cut_data)
                r_shape_y = pi.shape[0]
                r_shape_x = pi.shape[1]
                res_data = pi[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
                res_data = proceesing.normalize(res_data)
                res_data = proceesing.resize(res_data, 300)
                res_data = np.ascontiguousarray(res_data.reshape(1,300,300,3))
                no_nan_no_ring_list.append(res_data)
                
        stop = time.time()
        print((stop-start)/60)

        no_nan_no_ring_list.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)


