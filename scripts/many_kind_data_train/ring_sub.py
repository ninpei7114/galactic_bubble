import numpy as np
import pandas as pd
from skimage import transform
import copy

import label_caliculator
import proceesing
import astroquery.vizier


"""
リングのaugmentationパターンを作成するスクリプト
"""

def flip_data(source_data):
    r_shape_y = source_data.shape[0]
    r_shape_x = source_data.shape[1]

    ud = np.flipud(source_data)
    ud_res_data = ud[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
    ud_res_data = proceesing.normalize(ud_res_data)
    ud_res_data = proceesing.resize(ud_res_data, 300)



    lr = np.fliplr(source_data)
    lr_res_data = lr[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
    lr_res_data = proceesing.normalize(lr_res_data)
    lr_res_data = proceesing.resize(lr_res_data, 300)

    return ud_res_data, lr_res_data



def rotate_data(source_data, deg, xmin_list, ymin_list, xmax_list, ymax_list, name_list, fits_path):
    rotate_cut_data = transform.rotate(source_data, deg)
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


    r_shape_y = rotate_cut_data.shape[0]
    r_shape_x = rotate_cut_data.shape[1]
    res_data = rotate_cut_data[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
    res_data = proceesing.normalize(res_data)
    res_data = proceesing.resize(res_data, 300)

    info = {'fits':fits_path, 'name':name_list, 'xmin':xmin_list, 'xmax':xmax_list, 
                        'ymin':ymin_list, 'ymax':ymax_list}
    return res_data, rotate_cut_data, info




def scale(row, w,GLON_new_min,GLON_new_max, GLAT_min, GLAT_max, scale, star_dic, mode, MWP, data, fits_path):
    x_pix_min, y_pix_min, x_pix_max, y_pix_max, width, hight, flag = label_caliculator.calc_pix(row, w,GLON_new_min,GLON_new_max,
                                                                                    GLAT_min, GLAT_max, mode, scale)

    if flag: #calc_pix時に100回試行してもできなかった場合の場合分け
        cover_star_position, cover_star_name = label_caliculator.find_cover(star_dic, x_pix_min, y_pix_min, x_pix_max, y_pix_max)

        sig1 = 1/(2*(np.log(2))**(1/2))
        if x_pix_min<0 or y_pix_min<0:
#                 print('min_error')
            return False, 0, 0
        else:
            c_data = data[int(y_pix_min):int(y_pix_max), int(x_pix_min):int(x_pix_max)].view()
            cut_data = copy.deepcopy(c_data)
            # print(cut_data.shape)
            # print(x_pix_min, y_pix_min, x_pix_max, y_pix_max, width, hight)
            pi = proceesing.conv(300, sig1, cut_data)
            r_shape_y = pi.shape[0]
            r_shape_x = pi.shape[1]
            res_data = pi[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
            res_data = proceesing.normalize(res_data)
            res_data = proceesing.resize(res_data, 300)
            
            if np.isnan(res_data.sum()):
                return False, 0, 0
            else:
                xmin_list, ymin_list, xmax_list, ymax_list, name_list = label_caliculator.make_label(x_pix_min, y_pix_min, x_pix_max, y_pix_max, 
                                                                                                        cover_star_position, cover_star_name,
                                                                                                        width, hight, MWP)
                xmin_list, ymin_list, xmax_list, ymax_list = label_caliculator.check_list(xmin_list, ymin_list, 
                                                                                                            xmax_list, ymax_list)
                                                                                            
                info = {'fits':fits_path, 'name':name_list, 'xmin':xmin_list, 'xmax':xmax_list, 
                        'ymin':ymin_list, 'ymax':ymax_list}

                return True, res_data, info

    else:
        return False, 0, 0





def translation(MWP, fits_path, data, star_dic, row, world, GLON_min, GLON_max, GLAT_min, GLAT_max):
    import random
    ccc = 0
    ok = True

    while ok:
        random_num = 1/np.random.uniform(0.125, 0.8)
        lmax = row['GLON'] + random_num*1.5*row['Reff']/60
        bmin = row['GLAT'] - random_num*1.5*row['Reff']/60
        #右端
        lmin = row['GLON'] - random_num*1.5*row['Reff']/60
        bmax = row['GLAT'] + random_num*1.5*row['Reff']/60
        ccc += 1
        if GLON_min<=lmin and lmax<=GLON_max and GLAT_min<=bmin and bmax<=GLAT_max:
            ok = False
            flag = True
        if ccc>=100:
            ok = False
            flag = False

    x_min, y_min = world.all_world2pix(lmax, bmin, 0)
    x_max, y_max = world.all_world2pix(lmin, bmax, 0)
    r = int((x_max - x_min)/(2*random_num))#ringの半径pixel
    
    width = x_max - x_min
    height = y_max - y_min
    
    x_pix_min = x_min - width/2
    y_pix_min = y_min - height/2
    x_pix_max = x_max + width/2
    y_pix_max = y_max + height/2

    x_offset = random.uniform(-(random_num-0.5)*r, (random_num-0.5)*r)
    y_offset = random.uniform(-(random_num-0.5)*r, (random_num-0.5)*r)
    x_pix_min = x_pix_min + int(x_offset)
    x_pix_max = x_pix_max + int(x_offset)
    y_pix_min = y_pix_min + int(y_offset)
    y_pix_max = y_pix_max + int(y_offset)
    width = x_pix_max - x_pix_min
    height = y_pix_max - y_pix_min
    
    if flag: #calc_pix時に100回試行してもできなかった場合の場合分け   
        cover_star_position, cover_star_name = label_caliculator.find_cover(star_dic, x_pix_min, y_pix_min, x_pix_max, y_pix_max)
        c_data = data[int(y_pix_min):int(y_pix_max), int(x_pix_min):int(x_pix_max)].view()
        cut_data = copy.deepcopy(c_data)
        if np.isnan(cut_data.sum()):
            return flag, 0,  0
            
        else:
            sig1 = 1/(2*(np.log(2))**(1/2))
            pi = proceesing.conv(300, sig1, cut_data)
            xmin_list, ymin_list, xmax_list, ymax_list, name_list = label_caliculator.make_label(x_pix_min, y_pix_min, x_pix_max, y_pix_max, 
                                                                                cover_star_position, cover_star_name,
                                                                                width, height, MWP)
            r_shape_y = pi.shape[0]
            r_shape_x = pi.shape[1]
            res_data = pi[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
            res_data = proceesing.normalize(res_data)
            res_data = proceesing.resize(res_data, 300)
            xmin_list, ymin_list, xmax_list, ymax_list = label_caliculator.check_list(xmin_list, ymin_list, 
                                                                                        xmax_list, ymax_list)
            info = {'fits':fits_path, 'name':name_list, 'xmin':xmin_list, 'xmax':xmax_list, 
                        'ymin':ymin_list, 'ymax':ymax_list}
            return flag, res_data, info
    else:
        return flag, 0,  0



def catatologue(choice):
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
        print('this choice does not exist')


