import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage

import argparse
import astropy
import astropy.io.fits

import pathlib
import copy

import torch
from torch.nn import functional as F



def parse_args():
    parser = argparse.ArgumentParser(description='make data for deepcluster')

    parser.add_argument('path', metavar='DIR', help='path to dataset')
    parser.add_argument('--savedir', default='.', 
                        help='data save dir')

    return parser.parse_args()


def norm(data):
    min_ = np.min(data)
    b = np.std(data)
    mean = np.mean(data)
    data -= min_
    max_ = b*3 + mean
    data[data>max_] = max_

    data /= np.max(data)
    return data


def normalize(array):
    """
    入力：（y, x, 2 or 3）
    出力：（y ,x, 2 or 3）
    """
    gauss_list = []
    s = array.shape[2]
    for k in range(s):
        cut_data_k = array[:,:,k]
        cut_data_k = norm(cut_data_k)
        gauss_list.append(cut_data_k[:,:,None])                           

    cut_data = np.concatenate(gauss_list, axis=2)
    return cut_data


def resize(data, size):
    cut_data = np.swapaxes(data, 1, 2)
    cut_data = np.swapaxes(cut_data, 0, 1)
    cut_data = torch.from_numpy(cut_data)
    cut_data = cut_data.unsqueeze(0)
    resize_data = F.interpolate(cut_data, (size, size), mode='bilinear', align_corners=False)
    resize_data = resize_data.permute(0, 2, 3, 1)
    resize_data = np.squeeze(resize_data.detach().numpy())
    
    return resize_data


def conv(obj_size, obj_sig, data):
    """
    dataの入力サイズ↓
    入力：（y ,x, 2 or 3）
    出力：（size ,size, 2 or 3）
    -------------------------------
    切り出したデータがobj_sizeより大きければ、smoothingをする
    小さければ、そのまま返す。
    """
    if data.shape[0]/7*5>obj_size:
        fwhm = (data.shape[0]/7*5/obj_size)*2
        sig3 = fwhm/(2*(2*np.log(2))**(1/2))
        sig2 = (sig3**2-obj_sig**2)**(1/2)

        kernel = np.outer(signal.gaussian(8*round(sig2)+1, sig2), signal.gaussian(8*round(sig2)+1, sig2))
        kernel1= kernel/np.sum(kernel)

        conv_list = []
        for k in range(data.shape[2]):
            cut_data_k = data[:,:,k]
            lurred_k = signal.fftconvolve(cut_data_k, kernel1, mode='same')
            conv_list.append(lurred_k[:,:,None])

        pi = np.concatenate(conv_list, axis=2)
    else:
        pi = data
    return pi


def cut_data(data_, many_ind, cut_shape, sig1):
    data_list = []
    position_list_ = []
    for i in many_ind:
        ## データの余白を考えて、切り出し（convするときに生じる枠を後で取り除けるように）
        x_min = i[1]-cut_shape/5
        x_max = i[1]+cut_shape+cut_shape/5
        y_min = i[0]-cut_shape/5
        y_max = i[0]+cut_shape+cut_shape/5
        data_c = data_[int(y_min):int(y_max), int(x_min):int(x_max)].view()
#         print(int(y_min), int(y_max), int(x_min), int(x_max))
        ## nanの処理
        if np.max(data_c) == np.max(data_c):
            d = copy.deepcopy(data_c)
            d = conv(300, sig1, d)
            d = d[int(cut_shape/5):int(cut_shape*6/5), int(cut_shape/5):int(cut_shape*6/5)]
            
            flag = True
            for dim in range(d.shape[2]):
                ## データの中に0が何個あるかを計算、多ければ削除。　あまり意味はないかも
                non_zero_count = np.count_nonzero(d[:,:,dim])
                if non_zero_count>=d.shape[0]*d.shape[1]*3/4:
                    pass
                else:
                    flag = False
            if flag:
                d = normalize(d)
                d = resize(d, 300)
                data_list.append(d)
#                 position_list_.append([int(y_min)+int(cut_shape/5), int(x_min)+int(cut_shape/5)])
                position_list_.append([int(i[0]), int(i[1])])
        else:
            pass
    
    return data_list, position_list_


def remove_nan(data1):
    # fits???????????????max?????
    mask1_10 = (data1==data1)
    mask1_1010 = np.where(mask1_10, 0, 1)
    label1, name1 = scipy.ndimage.label(mask1_1010)
    data_areas1 = scipy.ndimage.sum(mask1_1010, label1, np.arange(name1+1))
    minsize1 = 1000
    data_mask1_10 =  (data_areas1 < minsize1)&(0 < data_areas1)
    small_mask1_10 = data_mask1_10[label1.ravel()].reshape(label1.shape)
    data1[small_mask1_10] = np.nanmax(data1)

    return data1


def main(args):

    # file_path = pathlib.Path('../../../../fits_data/remove_saturation_nan_fits/')
    sig1 = 1/(2*(np.log(2))**(1/2))
    spitzer_rfits = astropy.io.fits.open(pathlib.Path(args.path)/'r.fits')[0]
    spitzer_gfits = astropy.io.fits.open(pathlib.Path(args.path)/'g.fits')[0]

    data_ = np.concatenate([remove_nan(spitzer_rfits.data[:,:,None]), remove_nan(spitzer_gfits.data[:,:,None])], axis=2)
    
    size = 300
    cut_shape = (size, size)
    fragment = 2
    slide_pix = (int(round(cut_shape[0]/fragment)), int(round(cut_shape[1]/fragment)))
    
    shape = data_.shape
    x_num = int(shape[1]/slide_pix[1])-1
    y_num = int(shape[0]/slide_pix[0])-1

    x_idx = np.arange(cut_shape[1]/5, slide_pix[1]*x_num, slide_pix[1])
    y_idx = np.arange(cut_shape[0]/5, slide_pix[0]*y_num, slide_pix[0])
    x_ind, y_ind = np.meshgrid(x_idx, y_idx)
    
    l_ind = []
    for x, y in zip(x_ind.ravel(), y_ind.ravel()):
        l_ind.append([y, x])
    ind = np.array(l_ind)
    data_list, p_list = cut_data(data_, ind, cut_shape[0], sig1)
    p_data = np.array(data_list)
    # print(args.path.split('/')[-1])
    print(args.path.split('/')[-2])
    np.save('%s/%s_cut_300.npy'%(args.savedir, args.path.split('/')[-2]), p_data)


if __name__ == '__main__':
    args = parse_args()
    main(args)

