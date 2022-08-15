
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

from torchvision import transforms
import torch
from torch.nn import functional as F
from torch import nn




def GLON_LAT(data, header, w):
    
    a = data.shape[0]
    b = data.shape[1]
    GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
    GLON_max, GLAT_max = w.all_pix2world(0, a, 0)
    
# 主に銀中のfitsを扱う為の処理
# cut_no_ring関数でrandom uniformをするのだが、その時に大小関係をつける為に必要な処理
    if GLON_min>=358.0:
        GLON_min = GLON_min - 360
                
    else:
        pass
    
    GLON_center = (GLON_min + GLON_max)/2
    GLON_new_min = GLON_center - 1.5
    GLON_new_max = GLON_center + 1.5
    GLON_new_min1 = GLON_new_min + 3*18.670000076293945/60
    GLON_new_max1 = GLON_new_max - 3*18.670000076293945/60

    
    GLAT_center = (GLAT_min + GLAT_max)/2
    GLAT_new_min = GLAT_center - header[50]/2.1
    GLAT_new_max = GLAT_center + header[50]/2.1
    GLAT_new_min1 = GLAT_new_min + 3*18.670000076293945/60
    GLAT_new_max1 = GLAT_new_max - 3*18.670000076293945/60
#     return GLON_new_min, GLON_new_max, GLAT_new_min, GLAT_new_max
    return GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1



def calc_cut_pix(fits_data, GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1, hist_, hisy, range_, w):
    
    random_GLON = random.uniform(GLON_new_min1, GLON_new_max1)
    random_GLAT = random.uniform(GLAT_new_min1, GLAT_new_max1)
    q = np.random.choice(np.arange(0, len(hist_)), p = hisy)
    random_Rout = random.uniform(range_[q+1], range_[q])
    
    lmax_random = random_GLON + 3*random_Rout/60
    bmin_random = random_GLAT - 3*random_Rout/60
    #右端
     #銀中のfitsを扱うための処理
     # 
    if GLON_new_min1< 0:
        lmin_random = random_GLON - 3*random_Rout/60 +360
    
    else:
        lmin_random = random_GLON - 3*random_Rout/60 
        
    bmax_random = random_GLAT + 3*random_Rout/60
    ### おそらくworld2pixは360を超えても-360をしてくれる（今回は関係ないが）
    x_random_min, y_random_min = w.all_world2pix(lmax_random, bmin_random, 0)
    x_random_max, y_random_max = w.all_world2pix(lmin_random, bmax_random, 0)
#     print(x_random_min, x_random_max, y_random_min, y_random_max)
    
    width = x_random_max - x_random_min
    hight = y_random_max - y_random_min
#     print(width, hight)
    x_random_min = x_random_min - width/2
    x_random_max = x_random_max + width/2
    y_random_min = y_random_min - hight/2
    y_random_max = y_random_max + hight/2
#     print(x_random_min, x_random_max, y_random_min, y_random_max)
    
    return x_random_min, x_random_max, y_random_min, y_random_max



def cut_no_ring(fits_data, GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1, hist_, hisy, range_, w):
    
    x_random_min, x_random_max, y_random_min, y_random_max = calc_cut_pix(fits_data, 
                                                                          GLON_new_min1, GLON_new_max1, 
                                                                          GLAT_new_min1, GLAT_new_max1, hist_, hisy, range_, w)
    
    while x_random_min<=0 or x_random_max>fits_data.shape[1] or y_random_min<=0 or y_random_max>fits_data.shape[0]:
        
         x_random_min, x_random_max, y_random_min, y_random_max = calc_cut_pix(fits_data, 
                                                                          GLON_new_min1, GLON_new_max1, 
                                                                          GLAT_new_min1, GLAT_new_max1, hist_, hisy, range_, w)
          
    cut_data_random = fits_data[int(y_random_min):int(y_random_max), int(x_random_min):int(x_random_max)].view()
    cut_data_random_ = copy.deepcopy(cut_data_random)
    
    return cut_data_random_



#fitsの場所をnanにしたfitsを使う
def no_nan_ring(fits, GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1, hist_, hisy, range_, w):
    '''
    このすぐ下にcut_no_ring関数を使っている
    というのも、この関数はnan判定するための物だから
    '''

    cut_data_random = cut_no_ring(fits,GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1, hist_, hisy, range_, w)
    while np.isnan(np.sum(cut_data_random)):
        cut_data_random = cut_no_ring(fits,GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1, hist_, hisy, range_, w)
        
    else:
        pass
        
    return cut_data_random
### nan処理をする 
### cut_no_ring関数と併用するため、importする時はno_nan関数とcut_no_ring関数を同時にする必要がある
### no_nan関数で出てくるarrayは一つだけだから、for 文の中にこの関数を入れて使用する












