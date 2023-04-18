import astropy.io.fits
import astropy.wcs

import numpy as np
from numpy.random import default_rng
import pandas as pd

import copy
import os
import tqdm

from torch.nn import functional as F
import proceesing
import label_caliculator
import ring_sub



def make_ring(name, train_cfg, args, train_l):

    sig1 = 1/(2*(np.log(2))**(1/2))

    # choice catalogue from 'CH' or 'MWP'
    choice = 'CH'
    Ring_CATA = ring_sub.catalogue(choice)

    frame_mwp_train = []
    mwp_ring_list_train = []

    train_count = 0
    train_nan_count = 0
    pbar = tqdm.tqdm(range(len(train_l)))
    flip = train_cfg['flip']
    rot = train_cfg['rotate']
    scale = train_cfg['scale']
    translation = train_cfg['translation']
    trans_rg = default_rng(123)

    for i in pbar: 
        pbar.set_description(train_l[i])
        fits_path = train_l[i]
        spitzer_rfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'r.fits')[0]
        spitzer_gfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'g.fits')[0]
        spitzer_bfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'b.fits')[0]

        #RGBにしたいため、fitsのdataを重ねる
        data = np.concatenate([proceesing.remove_nan(spitzer_rfits.data[:,:,None]), 
                                proceesing.remove_nan(spitzer_gfits.data[:,:,None]), 
                                proceesing.remove_nan(spitzer_bfits.data[:,:,None])], axis=2)


        a = data.shape[0]
        b = data.shape[1]
        w = astropy.wcs.WCS(spitzer_rfits.header)
        GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
        GLON_max, GLAT_max = w.all_pix2world(0, a, 0) 

        GLON_center = (GLON_min+GLON_max)/2
        GLON_new_min = GLON_center-1.5
        GLON_new_max = GLON_center+1.5
        
        Ring_cata = Ring_CATA.query('@GLON_new_min < GLON <= @GLON_new_max')
        Ring_cata = Ring_cata.reset_index()
        train_count += len(Ring_cata)

        # star_listは辞書
        label_cal = label_caliculator.label_caliculator(choice, 'train', w)
        label_cal.all_star(Ring_cata)
        # print(fits_path)

        for _, row in Ring_cata.iterrows():    

            x_pix_min, y_pix_min, x_pix_max, y_pix_max, flag = label_cal.calc_pix(row, 
                                                                                GLON_new_min, GLON_new_max,
                                                                                GLAT_min, GLAT_max, 1/0.89)

            if flag: #calc_pix時に100回試行してもできなかった場合の場合分け   
                label_cal.find_cover()

                if x_pix_min<0 or y_pix_min<0:
                    pass

                else:
                    c_data = data[int(y_pix_min):int(y_pix_max), int(x_pix_min):int(x_pix_max)].view()
                    cut_data = copy.deepcopy(c_data)

                    if np.isnan(cut_data.sum()):
                        pass
                        
                    else:

                        ########################
                        ## 普通に切り出したリング ##
                        ########################

                        pi = proceesing.conv(300, sig1, cut_data)
                        pi_ = copy.deepcopy(pi)
                        label_cal.make_label(Ring_CATA)
                        r_shape_y = pi_.shape[0]
                        r_shape_x = pi_.shape[1]
                        res_data = pi_[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
                        res_data = proceesing.normalize(res_data)
                        res_data = proceesing.resize(res_data, 300)
                        xmin_list, ymin_list, xmax_list, ymax_list, name_list = label_cal.check_list()
                            
                        info = {'fits':fits_path, 'name':name_list, 'xmin':xmin_list, 'xmax':xmax_list, 
                                'ymin':ymin_list, 'ymax':ymax_list}

                        def append_data(data, info, data_list, frame):
                            if not np.isnan(data.sum()):
                                data_list.append(data)
                                frame.append(info)

                        # append_data(res_data, info, mwp_ring_list_train, frame_mwp_train)
                        data_proc = ring_sub.data_proccessing(pi, fits_path, choice, name_list, 
                                                            xmin_list, ymin_list, xmax_list, ymax_list)
                        
                        ########################
                        ## データの複数種類を作成 ##
                        ########################
                        for _ in range(args.augmentation_ratio):
                                
                            m2_size = trans_rg.uniform(0.125, 0.8)
                            ###### 並行移動 ######
                            if translation:                       
                                
                                fl, trans_data, trans_info = data_proc.translation(row, GLON_new_min, GLON_new_max,
                                                                    GLAT_min, GLAT_max, Ring_CATA, data, label_cal, m2_size, trans_rg)
                                ## データやlabelの作成に不備があれば、fl=False(例えば、xmin<0や、xmin=xmaxなど)
                                ## 問題がなければ、fl=True
                                if fl:
                                    append_data(trans_data, trans_info, mwp_ring_list_train, frame_mwp_train)

                            ###### 回転 ######
                            if rot:
                                if translation:
                                    pass
                                else:
                                    fl, trans_data, trans_info = data_proc.translation(row, GLON_new_min, GLON_new_max,
                                                                    GLAT_min, GLAT_max, Ring_CATA, data, label_cal, m2_size, trans_rg)
                                if fl:
                                    for deg in [90, 180, 270]:
                                        rot_data, rotate_info = data_proc.rotate_data(deg, trans_data, trans_info)
                                        append_data(rot_data, rotate_info, mwp_ring_list_train, frame_mwp_train)
                            
                            ###### 上下反転 ######
                            if flip:
                                if translation:
                                    pass
                                else:
                                    fl, trans_data, trans_info = data_proc.translation(row, GLON_new_min, GLON_new_max,
                                                                    GLAT_min, GLAT_max, Ring_CATA, data, label_cal, m2_size, trans_rg)
                                if fl:               
                                    data_proc_flip_rot = ring_sub.data_proccessing(
                                        trans_data, fits_path, choice, trans_info['name'], 
                                        trans_info['xmin'], trans_info['ymin'], 
                                        trans_info['xmax'], trans_info['ymax'])
                                    ud_res_data, lr_res_data, ud_info, lr_info = data_proc_flip_rot.flip_data()
                                    append_data(ud_res_data, ud_info, mwp_ring_list_train, frame_mwp_train)
                                    append_data(lr_res_data, lr_info, mwp_ring_list_train, frame_mwp_train)

                        
    frame_mwp_train = pd.DataFrame(frame_mwp_train)
    frame_mwp_train['id']  = [i for i in range(len(frame_mwp_train))]

    print('train_count  ',  train_count)
    print('train_nan_count  ',  train_nan_count)

    mwp_ring_list_train = np.array(mwp_ring_list_train).astype(np.float32)
    
    savedir_name = name
    if os.path.exists(savedir_name):
        pass
    else:
        os.mkdir(savedir_name)

    # mwp_ring_list_train_ = np.array(mwp_ring_list_train)
    mwp_ring_list_train_ = mwp_ring_list_train*255
    mwp_ring_list_train_ = np.uint8(mwp_ring_list_train_)
    if mwp_ring_list_train_.shape[0]>3000:
        slice = 10
    else:
        slice = 1
    proceesing.data_view_rectangl(25, mwp_ring_list_train_[::slice], frame_mwp_train[::slice]).save(savedir_name + '/train_ring.pdf')
    frame_mwp_train.to_csv(savedir_name + '/train_label.csv')

    print('train_Ring_num : ', len(mwp_ring_list_train))
    print('train_Ring_label_num : ', len(frame_mwp_train))

    return mwp_ring_list_train, frame_mwp_train


