import astropy.io.fits
import astropy.wcs

import numpy as np
# from npy_append_array import NpyAppendArray
from numpy.random import default_rng

import time
import pathlib
import os
import argparse
from tqdm import tqdm

from PIL import Image
import json

import proceesing
import NonRing_sub

"""
webdatasetを使用するための、Non-Ringのpng画像を作成する
さらに、json形式のlabelも作成する

example command:
python make_NonRing.py /workspace/fits_data/ring_to_circle_nan_fits -r True
"""

def parse_args():
    parser = argparse.ArgumentParser(description='make data for SSD')
    parser.add_argument('fits_path', metavar='DIR', help='path to Ring_to_circle_nan_fits')
    parser.add_argument('--each_region', '-r', action='store_true')
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')

    return parser.parse_args()



def main(args):
    
    ################################################
    ## 領域ごとに作るのか、デフォルトの領域で作るのか選択 ##
    ################################################

    if args.each_region:
        ## 各領域ごとにNon-Ringを作成する
        ## 'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため使用しない
        all_l = [
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

        mode_l = ['all']
        if os.path.exists('/workspace/NonRing_png/region_NonRing_png'):
            pass
        else:
            os.mkdir('/workspace/NonRing_png/region_NonRing_png')

    else:
        ## デフォルトの領域を用いて、Non-Ringを作成する
        ## 'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため使用しない
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

        mode_l = ['train', 'val']
    
    random_uni = default_rng(123)

    #####################
    ## Non-Ring作成開始 ##
    #####################

    for mode in mode_l:

        if mode == 'train':
            epoch = 100
            ref_path_list = train_l
            choice_num = len(train_l)-1
            choice_list = random_uni.integers(0, choice_num, epoch)
            iter = 60
        elif mode == 'val':
            epoch = 30
            ref_path_list = val_l
            choice_num = len(val_l)-1
            choice_list = random_uni.integers(0, choice_num, epoch)
            iter = 30
        elif mode == 'all':
            epoch = len(all_l)
            ref_path_list = all_l
            iter = 2000

        start = time.time()
        sig1 = 1/(2*(np.log(2))**(1/2))
        fits_path = pathlib.Path(args.fits_path)

        pbar = tqdm(range(epoch))
        for k in pbar:
            if args.each_region:
                path = ref_path_list[k]
                if os.path.exists('/workspace/NonRing_png/region_NonRing_png/%s'%path):
                    pass
                else:
                    os.mkdir('/workspace/NonRing_png/region_NonRing_png/%s'%path)
            else:
                path = ref_path_list[choice_list[k]]

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

            # GLON_LAT関数でGLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1を出す
            NonRing_sub_c.GLON_LAT(data, header)
            
            for i in range(iter):

                cut_data = NonRing_sub_c.no_nan_ring()
                pi = proceesing.conv(300, sig1, cut_data)
                r_shape_y = pi.shape[0]
                r_shape_x = pi.shape[1]
                res_data = pi[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
                res_data = proceesing.normalize(res_data)
                res_data = proceesing.resize(res_data, 300)
                pil_image = Image.fromarray(np.uint8(res_data*255))

                #########################
                ## 保存ディレクトリを選択 ##
                #########################

                if args.each_region:
                    ## 領域ごとに保存していく
                    pil_image.save('/workspace/NonRing_png/region_NonRing_png/%s/NonRing_%s.png'%(path, k*iter+i))
                    with open('/workspace/NonRing_png/region_NonRing_png/%s/NonRing_%s.json'%(path, k*iter+i), 'w') as f:
                        json.dump([], f, indent=4)
                else:
                    ## デフォルトフォルダに保存
                    pil_image.save('/workspace/NonRing_png/default_NonRing_png/%s/NonRing_%s.png'%(mode, k*iter+i))
                    with open('/workspace/NonRing_png/default_NonRing_png/%s/NonRing_%s.json'%(mode, k*iter+i), 'w') as f:
                        json.dump([], f, indent=4)
                
        stop = time.time()
        print((stop-start)/60)


if __name__ == '__main__':
    args = parse_args()
    main(args)


