import numpy as np
from numpy.random import default_rng
import copy


class NonRing_sub(object):

    def __init__(self, world, fits_data):

        self.hist = np.array([190., 138.,  97.,  55.,  29.,  10.,  12.,   8.,  12.,  11.,   4.,
            7.,   4.,   5.,   1.,   1.,   2.,   0.,   1.,   1.,   0.,   0.,
            1.,   0.,   0.,   0.,   1.,   0.,   0.,   1.])
        self.hist_ = self.hist/591
        self.hisy = self.hist_.tolist()
        self.range_ = np.array([ 0.11      ,  0.72866666,  1.3473333 ,  1.966     ,  2.5846667 ,
                3.2033334 ,  3.822     ,  4.4406667 ,  5.0593333 ,  5.678     ,
                6.2966666 ,  6.9153333 ,  7.534     ,  8.152667  ,  8.771334  ,
                9.39      , 10.008667  , 10.627334  , 11.246     , 11.864667  ,
                12.483334  , 13.102     , 13.720667  , 14.339334  , 14.958     ,
                15.576667  , 16.195333  , 16.814     , 17.432667  , 18.051332  ,
                18.67      ])

        self.w = world
        self.fits_data = fits_data
        self.fits_data_shape_x = fits_data.shape[1]
        self.fits_data_shape_y = fits_data.shape[0]
        self.random_uni = default_rng(123)

    def GLON_LAT(self, data, header):
        
        a = data.shape[0]
        b = data.shape[1]
        GLON_min, GLAT_min = self.w.all_pix2world(b, 0, 0)
        GLON_max, GLAT_max = self.w.all_pix2world(0, a, 0)
        
    # 主に銀中のfitsを扱う為の処理
    # cut_no_ring関数でrandom uniformをするのだが、その時に大小関係をつける為に必要な処理
        if GLON_min>=358.0:
            GLON_min = GLON_min - 360
                    
        else:
            pass
        
        GLON_center = (GLON_min + GLON_max)/2
        GLON_new_min = GLON_center - 1.5
        GLON_new_max = GLON_center + 1.5
        self.GLON_new_min1 = GLON_new_min + 3*18.670000076293945/60
        self.GLON_new_max1 = GLON_new_max - 3*18.670000076293945/60
        
        GLAT_center = (GLAT_min + GLAT_max)/2
        GLAT_new_min = GLAT_center - header[50]/2.1
        GLAT_new_max = GLAT_center + header[50]/2.1
        self.GLAT_new_min1 = GLAT_new_min + 3*18.670000076293945/60
        self.GLAT_new_max1 = GLAT_new_max - 3*18.670000076293945/60


    #     return GLON_new_min, GLON_new_max, GLAT_new_min, GLAT_new_max
        # return GLON_new_min1, GLON_new_max1, GLAT_new_min1, GLAT_new_max1



    def calc_cut_pix(self):
        
        random_GLON = self.random_uni.uniform(self.GLON_new_min1, self.GLON_new_max1)
        random_GLAT = self.random_uni.uniform(self.GLAT_new_min1, self.GLAT_new_max1)
        q = self.random_uni.choice(np.arange(0, len(self.hist_)), p = self.hisy)
        random_Rout = self.random_uni.uniform(self.range_[q+1], self.range_[q])
        
        lmax_random = random_GLON + 3*random_Rout/60
        bmin_random = random_GLAT - 3*random_Rout/60
        #右端
        #銀中のfitsを扱うための処理
        # 
        if self.GLON_new_min1< 0:
            lmin_random = random_GLON - 3*random_Rout/60 +360
        
        else:
            lmin_random = random_GLON - 3*random_Rout/60 
            
        bmax_random = random_GLAT + 3*random_Rout/60
        ### おそらくworld2pixは360を超えても-360をしてくれる（今回は関係ないが）
        x_random_min, y_random_min = self.w.all_world2pix(lmax_random, bmin_random, 0)
        x_random_max, y_random_max = self.w.all_world2pix(lmin_random, bmax_random, 0)
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



    def cut_no_ring(self):
        
        x_random_min, x_random_max, y_random_min, y_random_max = self.calc_cut_pix()
        
        while x_random_min<=0 or x_random_max>self.fits_data_shape_x or y_random_min<=0 or y_random_max>self.fits_data_shape_y:
            
            x_random_min, x_random_max, y_random_min, y_random_max = self.calc_cut_pix()
            
        cut_data_random = self.fits_data[int(y_random_min):int(y_random_max), int(x_random_min):int(x_random_max)].view()
        cut_data_random_ = copy.deepcopy(cut_data_random)
        
        return cut_data_random_



    #fitsの場所をnanにしたfitsを使う
    def no_nan_ring(self):
        '''
        このすぐ下にcut_no_ring関数を使っている
        というのも、この関数はnan判定するための物だから
        '''

        cut_data_random = self.cut_no_ring()
        while np.isnan(np.sum(cut_data_random)):
            cut_data_random = self.cut_no_ring()
            
        else:
            pass
            
        return cut_data_random
    ### nan処理をする 
    ### cut_no_ring関数と併用するため、importする時はno_nan関数とcut_no_ring関数を同時にする必要がある
    ### no_nan関数で出てくるarrayは一つだけだから、for 文の中にこの関数を入れて使用する












