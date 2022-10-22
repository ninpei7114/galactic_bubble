import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import pandas as pd

import pathlib
import cv2
import argparse

import astroquery.vizier
import astropy.wcs
import astropy.io.fits
import astropy



def parse_args():
    parser = argparse.ArgumentParser(description='make data for deepcluster')

    parser.add_argument('fits_path', metavar='DIR', help='path to dataset')
    parser.add_argument('save_dir', metavar='DIR', help='path to save directory')
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')

    return parser.parse_args()


def remove_nan(data1):
    """
    サチュレーションしている部分をmax値で埋める
    """
    
    mask1_10 = (data1==data1)
    mask1_1010 = np.where(mask1_10, 0, 1)
    label1, name1 = scipy.ndimage.label(mask1_1010)
    data_areas1 = scipy.ndimage.sum(mask1_1010, label1, np.arange(name1+1))
    minsize1 = 5000
    data_mask1_10 =  (data_areas1 < minsize1)&(0 < data_areas1)
    small_mask1_10 = data_mask1_10[label1.ravel()].reshape(label1.shape)
    data1[small_mask1_10] = np.nanmax(data1)

    return data1


def circle_nan(fits_data, same_shape_zero, pandas_catalog1, w):#, pandas_catalog2):
    for i in range(len(pandas_catalog1)):
    
        series_i = pandas_catalog1.iloc[i]
        #左端
        lmax = series_i['GLON'] + series_i['Rout']/60
        bmin = series_i['GLAT'] - series_i['Rout']/60
        #右端
        lmin = series_i['GLON'] - series_i['Rout']/60
        bmax = series_i['GLAT'] + series_i['Rout']/60
    
        x_pix_min, y_pix_min = w.all_world2pix(lmax, bmin, 0)
        x_pix_max, y_pix_max = w.all_world2pix(lmin, bmax,0)

        l = series_i['GLON']
        b = series_i['GLAT']
        lpix, bpix = w.all_world2pix(l, b, 0)
    
        rout = (x_pix_max-x_pix_min)/2
        same_shape_zero = cv2.circle(same_shape_zero,(int(np.round(lpix)), int(np.round(bpix))), int(rout),(255,255,255), -1)

    fits_data[same_shape_zero==same_shape_zero.max()]=np.nan
    
    return fits_data


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

    viz = astroquery.vizier.Vizier(columns=['*'])
    viz.ROW_LIMIT = -1
    bub_2006 = viz.query_constraints(catalog='J/ApJ/649/759/bubbles')[0].to_pandas()
    bub_2007 = viz.query_constraints(catalog='J/ApJ/670/428/bubble')[0].to_pandas()
    bub_2006_change = bub_2006.set_index('__CPA2006_')
    bub_2007_change = bub_2007.set_index('__CWP2007_')
    CH = pd.concat([bub_2006_change, bub_2007_change])
    CH.loc[CH['GLON']>=358.446500015535, 'GLON'] -=360 
    CH['CH'] = CH.index


    for i in l:
        print(i)
        for rgb in ['r.fits', 'g.fits', 'b.fits']:

            hdu = astropy.io.fits.open(args.fits_path/i/rgb)[0]
            header = hdu.header
            fits = remove_nan(hdu.data)

            w = astropy.wcs.WCS(header)
            a = fits.shape[0]
            b = fits.shape[1]

            GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
            GLON_max, GLAT_max = w.all_pix2world(0, a, 0)
            
            if GLON_min >=358.446500015535:
                GLON_min -= 360
            else:
                pass    
            
            cut_MWP = CH.query('@GLON_min < GLON < @GLON_max')
            c = np.zeros_like(fits)
            data = circle_nan(fits, c, cut_MWP, w)

            new_hdu = astropy.io.fits.PrimaryHDU(data, header)
            new_hdu_list = astropy.io.fits.HDUList([new_hdu])
            new_hdu_list.writeto(args.save_dir/i/rgb, overwrite=True)



if __name__ == '__main__':
    args = parse_args()
    main(args)