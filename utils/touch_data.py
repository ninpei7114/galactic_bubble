import matplotlib.pyplot as plt
import matlpotlib.patches as patches
from PIL import Image
import pandas as pd
import numpy
from scipy import signal

import pathlib
import copy

import astropy
import astropy.io.fits
import astropy.vizier
import astropy.wcs

import torch
from torch.nn import functional as F
from torch.nn import nn


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
    この関数は、上のnorm関数を使う
    チャンネルごとに規格化を行う
    ___________________________________
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
    """
    sizeは、自由　　　　　　
    今はy ,xは同じサイズだが、違うサイズにしたければ、タプルでsizeを入力するとよい
    入力データ：（y, x, 2 or 3）
    出力：（size ,size, 2 or 3）
    """
    cut_data = np.swapaxes(data, 1, 2)
    cut_data = np.swapaxes(cut_data, 0, 1)
    cut_data = torch.from_numpy(cut_data)
    cut_data = cut_data.unsqueeze(0)
    resize_data = F.interpolate(cut_data, (size, size), mode='bilinear', align_corners=False)
    resize_data = np.squeeze(resize_data.detach().numpy())

    resize_data_ = np.swapaxes(resize_data, 0, 1)
    resize_data_ = np.swapaxes(resize_data_, 1, 2)
    return resize_data_

def conv(obj_size, obj_sig, data):
    """
    入力：（y ,x, 2 or 3）
    出力：（size ,size, 2 or 3）
    -------------------------------
    切り出したデータがobj_sizeより大きければ、smoothingをする
    小さければ、そのまま返す。
    """
    if data.shape[0]>obj_size:
        fwhm = (data.shape[0]/obj_size)*2
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
