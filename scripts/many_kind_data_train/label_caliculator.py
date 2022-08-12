import astropy.io.fits
import astropy.wcs

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib.pyplot as plt
import random


import torch
from torch.nn import functional as F
from torch import nn




def find_cover(star_list, x_pix_min, y_pix_min, x_pix_max, y_pix_max):
    """
    切り出した画像の中に、他のリングが入っていないか確かめる。
    入っていたら、ラベル付けする
    star_listはdictionaryで、中身は、x_pix_min, y_pix_min, x_pix_max, y_pix_maxという順になっている
    """
    width = (x_pix_max - x_pix_min)/4
    hight = (y_pix_max - y_pix_min)/4
    
    g_area = ((x_pix_max-width)-(x_pix_min+width))*((y_pix_max-hight)-(y_pix_min+hight))
    
    overlapp_list = []
    overlapp_name = []
    for d in star_list.items():
        s_xmin = d[1][0]
        s_xmax = d[1][2]
        s_ymin = d[1][1]
        s_ymax = d[1][3]
        
        xx = np.array([s_xmin, s_xmax])
        yy = np.array([s_ymin, s_ymax])
        c_xx = np.clip(xx, x_pix_min+width, x_pix_max-width)
        c_yy = np.clip(yy, y_pix_min+hight, y_pix_max-hight)   
        s_area = (xx[1]-xx[0])*(yy[1]-yy[0])
        c_area = (c_xx[1]-c_xx[0])*(c_yy[1]-c_yy[0])
        
#         print('area : ', s_area, g_area*3/4)
        # 場合分け、全体に対してringが1/2以上入っていないといけない
        # 大きさが画像に対して、1/8以上でないとlabel付けしない
        if (c_area>=s_area*1/4 and (d[1][2]-d[1][0])>=(width*2)/8 and 
            (d[1][3]-d[1][1])>=(hight*2)/10):
            overlapp_list.append(d)
            overlapp_name.append(d[0])

        else:pass
        
    return overlapp_list, overlapp_name



def all_star(dataframe, world):
    """
    データセットのringの範囲をここで決める
    1.5倍で切り出し
    """
    star_dic = {}
    for index, row in dataframe.iterrows():    
    
        lmax = row['GLON'] + 1.5*row['Rout']/60
        bmin = row['GLAT'] - 1.5*row['Rout']/60
        #右端
        lmin = row['GLON'] - 1.5*row['Rout']/60
        bmax = row['GLAT'] + 1.5*row['Rout']/60
        #これは、リングを切り取る範囲　　切り取る範囲はRoutの3倍
        x_pix_min, y_pix_min = world.all_world2pix(lmax, bmin, 0)
        x_pix_max, y_pix_max = world.all_world2pix(lmin, bmax, 0)

        
        star_dic[row['MWP']] = [x_pix_min, y_pix_min, x_pix_max, y_pix_max]
        
    return star_dic



def calc_pix(row, world, GLON_min,GLON_max,GLAT_min,GLAT_max, mode):
    """
    切り出す画像の範囲をここで決める

    """
    import random

    ccc = 0
    ok = True
    

    while ok:
        if mode=='train':
            # random_num = 1/np.random.uniform(0.3, 0.89) #サイズが一様ver
            random_num = 1/0.89
        else:
            random_num = 1/0.89
        lmax = row['GLON'] + random_num*1.5*row['Rout']/60
        bmin = row['GLAT'] - random_num*1.5*row['Rout']/60
        #右端
        lmin = row['GLON'] - random_num*1.5*row['Rout']/60
        bmax = row['GLAT'] + random_num*1.5*row['Rout']/60
        ccc += 1
        if GLON_min<=lmin and lmax<=GLON_max and GLAT_min<=bmin and bmax<=GLAT_max:
            ok = False
            flag = True
        if ccc>=400:
            ok = False
            flag = False
        
    #これは、リングを切り取る範囲　　
    x_min, y_min = world.all_world2pix(lmax, bmin, 0)
    x_max, y_max = world.all_world2pix(lmin, bmax, 0)
    r = int((x_max - x_min)/(2*random_num))#ringの半径pixel
    
    width = x_max - x_min
    height = y_max - y_min
    
    x_pix_min = x_min - width/2
    y_pix_min = y_min - height/2
    x_pix_max = x_max + width/2
    y_pix_max = y_max + height/2
    
    #random_num - ２とは、切り出した画像が一辺random_num*rに対し、bboxが2*rだから、画像からリングがはみ出さないように
#     x_offset = random.uniform(-(random_num-1.5)*r, (random_num-1.5)*r)
#     y_offset = random.uniform(-(random_num-1.5)*r, (random_num-1.5)*r)
    # x_offset = random.uniform(-(random_num-1)*r-r/2, (random_num-1)*r+r/2)
    # y_offset = random.uniform(-(random_num-1)*r-r/2, (random_num-1)*r+r/2)
    # x_pix_min = x_pix_min + int(x_offset)
    # x_pix_max = x_pix_max + int(x_offset)
    # y_pix_min = y_pix_min + int(y_offset)
    # y_pix_max = y_pix_max + int(y_offset)
    width = x_pix_max - x_pix_min
    height = y_pix_max - y_pix_min
    
    
    return  x_pix_min, y_pix_min, x_pix_max, y_pix_max, width, height, flag#, star_list



def judge_01(number):
    if number > 1:
        return 1
    elif number<0:
        return 0
    else:
        return number




def make_label(x_pix_min, y_pix_min,x_pix_max, y_pix_max, cover_star_position, cover_star_name,  width, hight, MWP):
    """
    sは、主体となるringの位置情報
    x_pix_min, y_pix_min,x_pix_max, y_pix_maxは、切り出す画像のサイズ
    主体となるringに重なっているringのindex情報、重なったringの情報はstar_listの中にある。
    """

    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    named_list = []
    MWP_name_select = MWP.index.tolist()
    #切り出した画像にたまたま入った天体があるか、ないか
    if len(cover_star_position) == 0:
        pass
    else:
        
        for p, n in zip(cover_star_position, cover_star_name):
            # pは、('2G0020120-0068213', [array(7573.50002914), array(4663.19997904), 
             #                           array(7673.50003014), array(4763.19998004)])
            #のように、天体名とpostionが入っている
            if p[0] in MWP_name_select:
                
                xmin_c = p[1][0] - (x_pix_min+width/4)
                ymin_c = p[1][1] - (y_pix_min+hight/4)
                xmax_c = p[1][2] - (x_pix_min+width/4)
                ymax_c = p[1][3] - (y_pix_min+hight/4)
                xmin_list.append(judge_01(xmin_c/(width/2)))
                xmax_list.append(judge_01(xmax_c/(width/2)))
                ymin_list.append(judge_01(ymin_c/(hight/2)))
                ymax_list.append(judge_01(ymax_c/(hight/2)))
                named_list.append(n)
            
    return xmin_list, ymin_list, xmax_list, ymax_list, named_list




# def m_catalogue(sentei_path, mwp_catalogu_path):
#     nishimoto = pd.read_csv(sentei_path)
#     nishimoto = nishimoto.drop('Unnamed: 0', axis=1)
#     nishimoto = nishimoto.fillna(0)


#     rank1 = []
#     rank2 = []
#     rank3 = []
#     rank4 = []
#     rank5 = []
#     for i in range(len(nishimoto)):
#         nishimoto_s = nishimoto.loc[i]
#         Q = np.where(np.array(nishimoto_s.tolist())==1)[0]
#     #     print(Q)
#         if Q == 0:
#             rank1.append(i)
#         elif Q == 1:
#             rank2.append(i)
#         elif Q == 2:
#             rank3.append(i)
#         elif Q == 3:
#             rank4.append(i)
#         elif Q == 4:
#             rank5.append(i)

    
#     catalogue = pd.read_csv(mwp_catalogu_path)

#     each_rank = []
#     for i in range(len(nishimoto)):
#         nishimoto_s = nishimoto.loc[i]
#         Q = np.where(np.array(nishimoto_s.tolist())==1)[0][0]
#         each_rank.append(Q+1)

#     catalogue['rank'] = each_rank
#     catalogue_ = pd.concat([catalogue.iloc[rank3], catalogue.iloc[rank4], catalogue.iloc[rank5]])
#     catalogue_ = catalogue_.rename(columns={'Unnamed: 0':'MWP'})
#     MWP = catalogue_.set_index('MWP')

#     return MWP

