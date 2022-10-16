import numpy as np
import pandas as pd
from skimage import transform
import copy

import proceesing
import astroquery.vizier

from numpy.random import default_rng


"""
リングのaugmentationパターンを作成するスクリプト
"""

class data_proccessing(object):
    def __init__(self, source_data, fits_path, choice, name_list, xmin_list, ymin_list, xmax_list, ymax_list):

        self.source_data = source_data
        self.fits_path = fits_path
        self.choice = choice
        self.name_list = name_list
        self.xmin_list = xmin_list
        self.ymin_list = ymin_list 
        self.xmax_list = xmax_list 
        self.ymax_list = ymax_list

        if choice == 'MWP':
            self.Rout = 'Reff'
        else:
            self.Rout = 'Rout'

        self.rg = default_rng(123)



    def norm_res(self, data):
        """
        データを切り取り、
        normalizeとresizeをする。
        """
        shape_y = data.shape[0]
        shape_x = data.shape[1]
        data = data[int(shape_y/4):int(shape_y*3/4), int(shape_x/4):int(shape_x*3/4)]
        data = proceesing.normalize(data)
        data = proceesing.resize(data, 300)

        return data



    def flip_data(self):
        """
        リングを上下左右反転させる。
        ターゲットのリング以外の割り込みリングもあるため、
        flipもラベルの修正が必要
        """
        tempo = copy.deepcopy(self.source_data)
        ud = np.flipud(tempo)
        ud_res_data = self.norm_res(ud)
        
        ud_xmin_list, ud_xmax_list, ud_ymin_list, ud_ymax_list = [], [], [], []
        for i in range(len(self.xmin_list)):
            y_min = -(self.ymin_list[i] - 0.5) + 0.5
            x_min = self.xmin_list[i]
            y_max = -(self.ymax_list[i] - 0.5) + 0.5
            x_max = self.xmax_list[i]
            ud_xmin_list.append(x_min)
            ud_xmax_list.append(x_max)
            ud_ymin_list.append(y_min)
            ud_ymax_list.append(y_max)
        ud_info = {'fits':self.fits_path, 'name':self.name_list, 'xmin':ud_xmin_list, 'xmax':ud_xmax_list, 
                    'ymin':ud_ymin_list, 'ymax':ud_ymax_list}

        tempo = copy.deepcopy(self.source_data)
        lr = np.fliplr(tempo)
        lr_res_data = self.norm_res(lr)

        lr_xmin_list, lr_xmax_list, lr_ymin_list, lr_ymax_list = [], [], [], []
        for i in range(len(self.xmin_list)):
            x_min = -(self.xmin_list[i] - 0.5) + 0.5
            y_min = self.ymin_list[i]
            x_max = -(self.xmax_list[i] - 0.5) + 0.5
            y_max = self.ymax_list[i]
            lr_xmin_list.append(x_min)
            lr_xmax_list.append(x_max)
            lr_ymin_list.append(y_min)
            lr_ymax_list.append(y_max)
        lr_info = {'fits':self.fits_path, 'name':self.name_list, 'xmin':lr_xmin_list, 'xmax':lr_xmax_list, 
                    'ymin':lr_ymin_list, 'ymax':lr_ymax_list}

        return ud_res_data, lr_res_data, ud_info, lr_info




    def rotate_data(self, deg):
        """
        リングを回転させる。
        """
        tempo = copy.deepcopy(self.source_data)
        rotate_cut_data = transform.rotate(tempo, deg)
        xmin_list_, ymin_list_, xmax_list_, ymax_list_ = [], [], [], []

        for xy_num in range(len(self.xmin_list)):
            width = self.xmax_list[xy_num] - self.xmin_list[xy_num]
            center_x = ((self.xmin_list[xy_num] - 0.5) + (self.xmax_list[xy_num] - 0.5))/2
            center_y = ((self.ymin_list[xy_num] - 0.5) + (self.ymax_list[xy_num] - 0.5))/2

            new_center_x = center_x*np.cos(np.deg2rad(-deg)) - center_y*np.sin(np.deg2rad(-deg)) + 0.5
            new_center_y = center_x*np.sin(np.deg2rad(-deg)) + center_y*np.cos(np.deg2rad(-deg)) + 0.5

            xmin_list_.append(np.clip(new_center_x - width/2, 0, 1))
            ymin_list_.append(np.clip(new_center_y - width/2, 0, 1))
            xmax_list_.append(np.clip(new_center_x + width/2, 0, 1))
            ymax_list_.append(np.clip(new_center_y + width/2, 0, 1))

        res_data = self.norm_res(rotate_cut_data)
        info = {'fits':self.fits_path, 'name':self.name_list, 'xmin':xmin_list_, 'xmax':xmax_list_, 
                            'ymin':ymin_list_, 'ymax':ymax_list_}

        return res_data, info



    def scale(self, row, GLON_new_min, GLON_new_max, GLAT_min, GLAT_max, scale, MWP, data, label_cal):
        """
        サイズを縮小したパターンのRingを作り出す
        """
        x_pix_min, y_pix_min, x_pix_max, y_pix_max, flag = label_cal.calc_pix(row, GLON_new_min, GLON_new_max,
                                                                                        GLAT_min, GLAT_max, scale)
        #calc_pix時に100回試行してもできなかった場合の場合分け
        if flag: 
            label_cal.find_cover()

            sig1 = 1/(2*(np.log(2))**(1/2))
            if x_pix_min<0 or y_pix_min<0:

                return False, 0, 0
            else:
                c_data = data[int(y_pix_min):int(y_pix_max), int(x_pix_min):int(x_pix_max)].view()
                cut_data = copy.deepcopy(c_data)
                pi = proceesing.conv(300, sig1, cut_data)
                res_data = self.norm_res(pi)
                
                if np.isnan(res_data.sum()):
                    return False, 0, 0
                else:
                    label_cal.make_label(MWP)
                    xmin_list, ymin_list, xmax_list, ymax_list, name_list = label_cal.check_list()
                                                                                                
                    info = {'fits':self.fits_path, 'name':name_list, 'xmin':xmin_list, 'xmax':xmax_list, 
                            'ymin':ymin_list, 'ymax':ymax_list}

                    return True, res_data, info

        else:
            return False, 0, 0


    def translation(self, row, GLON_new_min, GLON_new_max, GLAT_min, GLAT_max, MWP, data, label_cal):
        random_num = 1/self.rg.uniform(0.125, 0.8)

        x_pix_min, y_pix_min, x_pix_max, y_pix_max, flag = label_cal.calc_pix(row, GLON_new_min, GLON_new_max,
                                                                                        GLAT_min, GLAT_max, random_num)
            
        half_width = (x_pix_max - x_pix_min)/4
        r = int(((x_pix_max - half_width) - (x_pix_min + half_width))/(2*random_num))
        x_offset = self.rg.uniform(-(random_num-0.5)*r, (random_num-0.5)*r)
        y_offset = self.rg.uniform(-(random_num-0.5)*r, (random_num-0.5)*r)
        x_pix_min = x_pix_min + int(x_offset)
        x_pix_max = x_pix_max + int(x_offset)
        y_pix_min = y_pix_min + int(y_offset)
        y_pix_max = y_pix_max + int(y_offset)
        width = x_pix_max - x_pix_min
        height = y_pix_max - y_pix_min
        
        if flag: #calc_pix時に100回試行してもできなかった場合の場合分け   
            label_cal.find_cover_for_translation(x_pix_min, y_pix_min, x_pix_max, y_pix_max)

            c_data = data[int(y_pix_min):int(y_pix_max), int(x_pix_min):int(x_pix_max)].view()
            cut_data = copy.deepcopy(c_data)
            if np.isnan(cut_data.sum()):
                return flag, 0,  0
                
            else:
                sig1 = 1/(2*(np.log(2))**(1/2))
                pi = proceesing.conv(300, sig1, cut_data)
                xmin_list, ymin_list, xmax_list, ymax_list, name_list = label_cal.make_label_for_translation(x_pix_min, y_pix_min, x_pix_max, y_pix_max, 
                                                                                    width, height, MWP)
                r_shape_y = pi.shape[0]
                r_shape_x = pi.shape[1]
                res_data = pi[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
                res_data = proceesing.normalize(res_data)
                res_data = proceesing.resize(res_data, 300)
                xmin_list, ymin_list, xmax_list, ymax_list = label_cal.check_list(xmin_list, ymin_list, 
                                                                                            xmax_list, ymax_list)
                info = {'fits':self.fits_path, 'name':name_list, 'xmin':xmin_list, 'xmax':xmax_list, 
                            'ymin':ymin_list, 'ymax':ymax_list}
                return flag, res_data, info
        else:
            return flag, 0,  0



def catalogue(choice):
    if choice=='CH':
        viz = astroquery.vizier.Vizier(columns=['*'])
        viz.ROW_LIMIT = -1
        bub_2006 = viz.query_constraints(catalog='J/ApJ/649/759/bubbles')[0].to_pandas()
        bub_2007 = viz.query_constraints(catalog='J/ApJ/670/428/bubble')[0].to_pandas()
        bub_2006_change = bub_2006.set_index('__CPA2006_')
        bub_2007_change = bub_2007.set_index('__CWP2007_')
        CH = pd.concat([bub_2006_change, bub_2007_change])
        CH['CH'] = CH.index

        return CH

    elif choice=='MWP':
        viz = astroquery.vizier.Vizier(columns=['*'])
        viz.ROW_LIMIT = -1
        MWP = viz.query_constraints(catalog='2019yCat..74881141J ')[0].to_pandas()
        MWP.loc[MWP['GLON']>=358.446500015535, 'GLON'] -=360 
        return MWP

    else:
        print('this choice catalogu does not exist')


