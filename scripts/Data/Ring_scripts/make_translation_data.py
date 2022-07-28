import astropy.io.fits
import astroquery.vizier
import astropy.wcs
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import copy
import os
from torch.nn import functional as F
import proceesing
import label_caliculator
import pathlib



def parse_args():
    parser = argparse.ArgumentParser(description='make data for deepcluster')

    parser.add_argument('spitzer_path', metavar='DIR', help='path to dataset')
    # parser.add_argument('ring_sentei_path', metavar='DIR', help='path to ring setntei file')
    # parser.add_argument('mwp_catalogu_path', metavar='DIR', help='path to mwp catalogue')
    # parser.add_argument('augmentation_num', metavar='DIR', help='the number of augmentation')
    parser.add_argument('savedir_format', metavar='DIR', help='data save dir name format')
    parser.add_argument('--augmentation_num_list', required=True, nargs="*", type=int, help='the number of augmentation')
    # parser.add_argument('--savedir', default='.', 
    #                     help='data save dir')

    return parser.parse_args()



def main(args):

    val_l = ['spitzer_00900+0000_rgb','spitzer_03900+0000_rgb','spitzer_31200+0000_rgb','spitzer_34200+0000_rgb',
         'spitzer_33900+0000_rgb',]
    val_l = sorted(val_l)

    train_l = [
    'spitzer_02100+0000_rgb','spitzer_04200+0000_rgb','spitzer_33300+0000_rgb','spitzer_35400+0000_rgb',
    'spitzer_00300+0000_rgb','spitzer_02400+0000_rgb','spitzer_04500+0000_rgb','spitzer_31500+0000_rgb',
    'spitzer_33600+0000_rgb','spitzer_35700+0000_rgb','spitzer_00600+0000_rgb','spitzer_02700+0000_rgb',
    'spitzer_04800+0000_rgb','spitzer_29700+0000_rgb','spitzer_31800+0000_rgb','spitzer_03000+0000_rgb',
    'spitzer_05100+0000_rgb','spitzer_30000+0000_rgb','spitzer_32100+0000_rgb','spitzer_01200+0000_rgb',
    'spitzer_03300+0000_rgb','spitzer_05400+0000_rgb','spitzer_30300+0000_rgb','spitzer_32400+0000_rgb',
    'spitzer_34500+0000_rgb','spitzer_01500+0000_rgb','spitzer_03600+0000_rgb','spitzer_05700+0000_rgb',
    'spitzer_30600+0000_rgb','spitzer_32700+0000_rgb','spitzer_34800+0000_rgb','spitzer_01800+0000_rgb',
    'spitzer_06000+0000_rgb','spitzer_30900+0000_rgb','spitzer_33000+0000_rgb','spitzer_35100+0000_rgb']
    train_l = sorted(train_l)

    sig1 = 1/(2*(np.log(2))**(1/2))

    # MWP = label_caliculator.m_catalogue(args.ring_sentei_path, args.mwp_catalogu_path)

    viz = astroquery.vizier.Vizier(columns=['*'])
    viz.ROW_LIMIT = -1
    bub_2006 = viz.query_constraints(catalog='J/ApJ/649/759/bubbles')[0].to_pandas()
    bub_2007 = viz.query_constraints(catalog='J/ApJ/670/428/bubble')[0].to_pandas()
    bub_2006_change = bub_2006.set_index('__CPA2006_')
    bub_2007_change = bub_2007.set_index('__CWP2007_')
    MWP = pd.concat([bub_2006_change, bub_2007_change])
    MWP['MWP'] = MWP.index

    # viz = astroquery.vizier.Vizier(columns=['*'])
    # viz.ROW_LIMIT = -1
    # MWP = viz.query_constraints(catalog='2019yCat..74881141J ')[0].to_pandas()
    MWP.loc[MWP['GLON']>=358.446500015535, 'GLON'] -=360

    for ag in args.augmentation_num_list:
        frame_mwp_train = pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])
        mwp_ring_list_train = []

        frame_mwp_val = pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])
        mwp_ring_list_val = []
        
        for mode in ['train', 'val']:
            if mode == 'train':
                l = train_l
                train_count = 0
                train_nan_count = 0
            else:
                l = val_l
                val_count = 0
                val_nan_count = 0

            for i in range(len(l)): 

                fits_path = l[i]
                spitzer_rfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'r.fits')[0]
                spitzer_gfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'g.fits')[0]
                spitzer_bfits = astropy.io.fits.open(args.spitzer_path+'/'+fits_path+'/'+'b.fits')[0]

                #RGBにしたいため、fitsのdataを重ねる
                data = np.concatenate([proceesing.remove_nan(spitzer_rfits.data[:,:,None]), 
                                    proceesing.remove_nan(spitzer_gfits.data[:,:,None]), 
                                    proceesing.remove_nan(spitzer_bfits.data[:,:,None])], axis=2)


                a = data.shape[0]
                b = data.shape[1]
        #         data[data!=data] = 0
                w = astropy.wcs.WCS(spitzer_rfits.header)
                GLON_min, GLAT_min = w.all_pix2world(b, 0, 0)
                GLON_max, GLAT_max = w.all_pix2world(0, a, 0) 

                GLON_center = (GLON_min+GLON_max)/2
                GLON_new_min = GLON_center-1.5
                GLON_new_max = GLON_center+1.5

                mwp = MWP.query('@GLON_new_min < GLON <= @GLON_new_max')
                mwp = mwp.reset_index()
                if mode == 'train':
                    train_count += len(mwp)
                else:
                    val_count += len(mwp)
                # star_listは辞書
                star_dic = label_caliculator.all_star(mwp, w)
                print(fits_path)

                for _, row in mwp.iterrows():    
                    # rank = row['rank']
        #             print(rank)
        #             print(type(rank))
                    s = star_dic[row['MWP']] # 主体となるringの位置情報
                    if mode == 'train':
                        # if rank == 3:
                        #     augmentation_num = 1
                        # else:
                        augmentation_num = ag
                    else:
                        augmentation_num = 1
                    
                    for _ in range(int(augmentation_num)):

                        x_pix_min, y_pix_min, x_pix_max, y_pix_max, width, hight, flag = label_caliculator.calc_pix(row, w,GLON_new_min,GLON_new_max,
                                                                                        GLAT_min,GLAT_max, 5, mode)
                        
                        if flag: #calc_pix時に100回試行してもできなかった場合の場合分け
                            
                            cover_star_position, cover_star_name = label_caliculator.find_cover(star_dic, x_pix_min, y_pix_min, x_pix_max, y_pix_max)

                            if x_pix_min<0 or y_pix_min<0:

                                pass

                            else:
                                c_data = data[int(y_pix_min):int(y_pix_max), int(x_pix_min):int(x_pix_max)].view()
                                cut_data = copy.deepcopy(c_data)

                                if np.isnan(cut_data.sum()):
                                    
                                    # if mode == 'train':
                                    #     train_nan_count += 1
                                    #     pass
                                    # else:
                                    #     val_nan_count += 1
                                    pass
                                    
                                else:
                                    pi = proceesing.conv(300, sig1, cut_data)
                                    xmin_list, ymin_list, xmax_list, ymax_list, name_list = label_caliculator.make_label(x_pix_min, y_pix_min, x_pix_max, y_pix_max, 
                                                                                                    cover_star_position, cover_star_name,
                                                                                                    width, hight, MWP)
                            
                                    if mode == 'train':
                                        
                                        r_shape_y = pi.shape[0]
                                        r_shape_x = pi.shape[1]
                                        res_data = pi[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
                                        res_data = proceesing.normalize(res_data)
                                        res_data = proceesing.resize(res_data, 300)
                                        if np.isnan(res_data.sum()):
                                            pass
                                        else:
                                            info = [[fits_path, name_list, xmin_list, xmax_list, ymin_list, ymax_list]]
                                            p_data = pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax'], data=info)
                                            
                                            mwp_ring_list_train.append(res_data)
                                            frame_mwp_train = pd.concat([frame_mwp_train, p_data])
    
                                    
                                    if mode == 'val':
                                        r_shape_y = pi.shape[0]
                                        r_shape_x = pi.shape[1]
                                        res_data = pi[int(r_shape_y/4):int(r_shape_y*3/4), int(r_shape_x/4):int(r_shape_x*3/4)]
                                        res_data = proceesing.normalize(res_data)
                                        res_data = proceesing.resize(res_data, 300)
                                        if np.isnan(res_data.sum()):
                                            pass
                                        else:
                                            info = [[fits_path, name_list, xmin_list, xmax_list, ymin_list, ymax_list]]
                                            p_data = pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax'], data=info)
                                            
                                            mwp_ring_list_val.append(res_data)
                                            frame_mwp_val = pd.concat([frame_mwp_val, p_data])
        


        ## データ作成
        frame_mwp_train['id']  = [i for i in range(len(frame_mwp_train))]
        frame_mwp_val['id']  = [i for i in range(len(frame_mwp_val))]

        print('train_count  ',  train_count)
        print('val_count  ',  val_count)
        print('train_nan_count  ',  train_nan_count)
        print('val_nan_count  ',  val_nan_count)

        mwp_ring_list_train = np.array(mwp_ring_list_train).astype(np.float32)
        mwp_ring_list_val = np.array(mwp_ring_list_val).astype(np.float32)
        
        savedir_name = args.savedir_format + '_' + str(ag)

        if os.path.exists(savedir_name):
             pass
        else:
            os.mkdir(savedir_name)

        np.save(savedir_name + '/ring/train_ring_data.npy', mwp_ring_list_train)
        frame_mwp_train.to_csv(savedir_name + '/ring/train_ring_label.csv')
        np.save(savedir_name + '/ring/val_ring_data.npy', mwp_ring_list_val)
        frame_mwp_val.to_csv(savedir_name + '/ring/val_ring_label.csv')

        # mwp_ring_list_train_ = np.array(mwp_ring_list_train)
        mwp_ring_list_train_ = mwp_ring_list_train*255
        mwp_ring_list_train_ = np.uint8(mwp_ring_list_train_)
        proceesing.data_view_rectangl(25, mwp_ring_list_train_[::10], frame_mwp_train[::10]).save(savedir_name + '/ring/train_ring.pdf')

        mwp_ring_list_val_ = mwp_ring_list_val*255
        mwp_ring_list_val_ = np.uint8(mwp_ring_list_val_)
        proceesing.data_view_rectangl(25, mwp_ring_list_val_, frame_mwp_val).save(savedir_name + '/ring/val_ring.pdf')

        width = []
        for i in range(len(frame_mwp_train)):
            row = frame_mwp_train.iloc[i]
            for k in range(len(row['xmin'])):
                xmin = 300*row['xmin'][k]
                xmax = 300*row['xmax'][k]
                ymin = 300*row['ymin'][k]
                ymax = 300*row['ymax'][k]
                
                width.append(xmax-xmin)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(width, bins=100)
        ax.set_xlabel('width_pix')
        ax.set_ylabel('number')
        fig.savefig(savedir_name + '/ring/width.png')


        x_cen = []
        y_cen = []
        for i in range(len(frame_mwp_train)):
            row = frame_mwp_train.iloc[i]
            for k in range(len(row['xmin'])):
                xmin = 300*row['xmin'][k]
                xmax = 300*row['xmax'][k]
                ymin = 300*row['ymin'][k]
                ymax = 300*row['ymax'][k]
                
                x_cen.append((xmax+xmin)/2)
                y_cen.append((ymax+ymin)/2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_cen, y_cen, s=0.05)
        plt.axes().set_aspect('equal') 
        ax.set_xlabel('x center')
        ax.set_ylabel('y center')

        fig.savefig(savedir_name + '/ring/center.png')



if __name__ == '__main__':
    args = parse_args()
    main(args)