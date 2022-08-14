import proceesing
import numpy as np
import pandas as pd
from skimage import transform
import proceesing
import label_caliculator
import ring_sub
import copy


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

    info = [[fits_path, name_list, xmin_list_, xmax_list_, ymin_list_, ymax_list_]]
    p_data = pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax'], data=info)

    return res_data, rotate_cut_data, p_data




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
                                                                                            
                info = [[fits_path, name_list, xmin_list, xmax_list, ymin_list, ymax_list]]
                p_data = pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax'], data=info)

                return True, res_data, p_data

    else:
        return False, 0, 0





