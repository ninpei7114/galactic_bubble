import numpy as np
import pandas as pd
from skimage import transform
import copy

import sys 
sys.path.append('../')
import proceesing
import astroquery.vizier


"""
リングのaugmentationパターンを作成するスクリプト
"""

class data_proccessing(object):
    def __init__(self, source_data, fits_path, choice):

        self.source_data = source_data
        self.fits_path = fits_path
        self.choice = choice
        if choice == 'MWP':
            self.Rout = 'Reff'
        else:
            self.Rout = 'Rout'

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
        """
        ud = np.flipud(self.source_data)
        ud_res_data = self.norm_res(ud)

        lr = np.fliplr(self.source_data)
        lr_res_data = self.norm_res(lr)

        return ud_res_data, lr_res_data


    def rotate_data(self, deg, xmin_list, ymin_list, xmax_list, ymax_list, name_list):
        """
        リングを回転させる。
        """
        rotate_cut_data = transform.rotate(self.source_data, deg)
        xmin_list_, ymin_list_, xmax_list_, ymax_list_ = [], [], [], []

        for xy_num in range(len(xmin_list)):
            width = xmax_list[xy_num] - xmin_list[xy_num]
            center_x = ((xmin_list[xy_num] - 0.5) + (xmax_list[xy_num] - 0.5))/2
            center_y = ((ymin_list[xy_num] - 0.5) + (ymax_list[xy_num] - 0.5))/2

            new_center_x = center_x*np.cos(np.deg2rad(-deg)) - center_y*np.sin(np.deg2rad(-deg)) + 0.5
            new_center_y = center_x*np.sin(np.deg2rad(-deg)) + center_y*np.cos(np.deg2rad(-deg)) + 0.5

            xmin_list_.append(np.clip(new_center_x - width/2, 0, 1))
            ymin_list_.append(np.clip(new_center_y - width/2, 0, 1))
            xmax_list_.append(np.clip(new_center_x + width/2, 0, 1))
            ymax_list_.append(np.clip(new_center_y + width/2, 0, 1))

        res_data = self.norm_res(rotate_cut_data)
        info = {'fits':self.fits_path, 'name':name_list, 'xmin':xmin_list, 'xmax':xmax_list, 
                            'ymin':ymin_list, 'ymax':ymax_list}

        return res_data, info


    def scale(self, row, w,GLON_new_min, GLON_new_max, GLAT_min, GLAT_max, scale, MWP, data, label_cal):
        """
        サイズ縮小
        """

        x_pix_min, y_pix_min, x_pix_max, y_pix_max, flag = label_cal.calc_pix(row, w,GLON_new_min, GLON_new_max,
                                                                                        GLAT_min, GLAT_max, scale)

        if flag: #calc_pix時に100回試行してもできなかった場合の場合分け
            label_cal.find_cover()

            sig1 = 1/(2*(np.log(2))**(1/2))
            if x_pix_min<0 or y_pix_min<0:
    #                 print('min_error')
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
                    xmin_list, ymin_list, xmax_list, ymax_list, name_list, star_dic = label_cal.check_list()
                                                                                                
                    info = {'fits':self.fits_path, 'name':name_list, 'xmin':xmin_list, 'xmax':xmax_list, 
                            'ymin':ymin_list, 'ymax':ymax_list}

                    return True, res_data, info

        else:
            return False, 0, 0



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


