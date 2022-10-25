import astropy
import astropy.io.fits
import aplpy
import astroquery.vizier

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import argparse
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description='make data for deepcluster')

    parser.add_argument('fits_path', metavar='DIR', help='path to dataset')
    parser.add_argument('save_dir', metavar='DIR', help='path to save directory')
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')

    return parser.parse_args()



def main(args):

    l = ['spitzer_02100+0000_rgb','spitzer_04200+0000_rgb','spitzer_33300+0000_rgb','spitzer_35400+0000_rgb','spitzer_00300+0000_rgb',
     'spitzer_02400+0000_rgb','spitzer_04500+0000_rgb','spitzer_31500+0000_rgb','spitzer_33600+0000_rgb','spitzer_35700+0000_rgb',
     'spitzer_00600+0000_rgb','spitzer_02700+0000_rgb','spitzer_04800+0000_rgb','spitzer_29700+0000_rgb','spitzer_31800+0000_rgb',
     'spitzer_03000+0000_rgb','spitzer_05100+0000_rgb','spitzer_30000+0000_rgb','spitzer_32100+0000_rgb','spitzer_01200+0000_rgb',
     'spitzer_03300+0000_rgb','spitzer_05400+0000_rgb','spitzer_30300+0000_rgb','spitzer_32400+0000_rgb','spitzer_34500+0000_rgb',
     'spitzer_01500+0000_rgb','spitzer_03600+0000_rgb','spitzer_05700+0000_rgb','spitzer_30600+0000_rgb','spitzer_32700+0000_rgb',
     'spitzer_34800+0000_rgb','spitzer_01800+0000_rgb','spitzer_06000+0000_rgb','spitzer_30900+0000_rgb','spitzer_33000+0000_rgb',
     'spitzer_35100+0000_rgb','spitzer_00900+0000_rgb','spitzer_03900+0000_rgb','spitzer_31200+0000_rgb','spitzer_34200+0000_rgb',
     'spitzer_33900+0000_rgb','spitzer_29400+0000_rgb','spitzer_06300+0000_rgb','spitzer_00000+0000_rgb']

    l = sorted(l)
    pbar = tqdm(range(len(l)))
    for i in range(len(l)):

        path = l[i]
        pbar.set_description(path)
        data_fits_R = args.fits_path + '/%s/r.fits'%path##2D
        data_fits_G = args.fits_path + '/%s/g.fits'%path##2D
        data_fits_B = args.fits_path + '/%s/b.fits'%path
        
        color_min_R = 10
        color_max_R = 100.0
        color_min_G = 10
        color_max_G = 130
        color_min_B = 30
        color_max_B = 100
        
        colorval = "%.1f_%.1f_%.1f_%.1f_%.1f_%.1f"%(color_min_R, color_max_R, color_min_G, color_max_G, color_min_B, color_max_B)
        if os.path.exists(args.save_dir + '/spitzer_aplpy'):
            pass
        else:
            os.mkdir(args.save_dir + '/spitzer_aplpy')

        save_png_name = 'spitzer_aplpy/RG_%s'%colorval + '_%s.png'%path
        fitss = [data_fits_R, data_fits_G, data_fits_B]
        
        aplpy.make_rgb_image(fitss, save_png_name, 
                        vmin_r=color_min_R, 
                        vmax_r=color_max_R,
                        vmin_g=color_min_G,
                        vmax_g=color_max_G, 
                        vmin_b=color_min_B, 
                        vmax_b=color_max_B, 
                        stretch_r="linear", 
                        stretch_g="linear", 
                        stretch_b="linear")



if __name__ == '__main__':
    args = parse_args()
    main(args)