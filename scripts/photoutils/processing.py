import copy

import cv2
import numpy as np
import scipy.ndimage
import torch
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy import signal
from torch.nn import functional as F


def norm_rp(data, nan_data_dim):
    data_min = np.nanmin(nan_data_dim)
    std = np.nanstd(nan_data_dim)
    mean = np.nanmean(nan_data_dim)
    data -= data_min
    max_ = std * 3 + mean

    data[data > max_] = max_
    data /= max_
    return data


def normalize_rp(array):
    """
    入力: (y, x, 2 or 3)
    出力: (y ,x, 2 or 3)
    """
    gauss_list = []
    dim = array.shape[2]
    nan_data = remove_peak(array)
    for k in range(dim):
        cut_data_k = array[:, :, k]
        cut_data_k = norm_rp(cut_data_k, nan_data[:, :, k])
        gauss_list.append(cut_data_k[:, :, None])
    cut_data = np.concatenate(gauss_list, axis=2)

    return cut_data


def remove_peak(array):
    data = array.copy()
    data_8micron = data[:, :, 1].copy()
    mean, median, std = sigma_clipped_stats(data_8micron, sigma=3)
    daofind = DAOStarFinder(fwhm=1.98, threshold=5 * mean)
    sources = daofind(data_8micron)
    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))

    same_shape_zero = np.zeros_like(data)
    for y, x in positions:
        same_shape_zero = cv2.circle(same_shape_zero, (int(y), int(x)), int(5), (255, 255, 255), -1)

    data[same_shape_zero == same_shape_zero.max()] = np.nan
    return data


def resize(data, size):
    """
    sizeは、自由
    今はy ,xは同じサイズだが、違うサイズにしたければ、タプルでsizeを入力するとよい
    入力データ:（y, x, 2 or 3）
    出力:（size ,size, 2 or 3）
    """
    cut_data = np.swapaxes(data, 1, 2)
    cut_data = np.swapaxes(cut_data, 0, 1)
    cut_data = torch.from_numpy(cut_data)
    cut_data = cut_data.unsqueeze(0)
    resize_data = F.interpolate(cut_data, (size, size), mode="bilinear", align_corners=False)
    resize_data = np.squeeze(resize_data.detach().numpy())

    resize_data_ = np.swapaxes(resize_data, 0, 1)
    resize_data_ = np.swapaxes(resize_data_, 1, 2)
    return resize_data_


def norm_res(data):
    """データを切り取り、normalizeとresizeをする。

    Args:
        data (numpy array): convolutionした生データ

    Returns:
        _type_: _description_
    """
    data_ = copy.deepcopy(data)
    data_ = normalize_rp(data_)
    data_ = resize(data_, 300)

    return data_


def conv(obj_size, obj_sig, data):
    """
    dataの入力サイズ↓
    入力:（y ,x, 2 or 3）
    出力:（size ,size, 2 or 3）
    -------------------------------
    切り出したデータがobj_sizeより大きければ、smoothingをする
    小さければ、そのまま返す。
    """

    if data.shape[0] > obj_size:
        fwhm = (data.shape[0] / obj_size) * 2
        sig3 = fwhm / (2 * (2 * np.log(2)) ** (1 / 2))
        sig2 = (sig3**2 - obj_sig**2) ** (1 / 2)

        kernel = np.outer(signal.gaussian(8 * round(sig2) + 1, sig2), signal.gaussian(8 * round(sig2) + 1, sig2))
        kernel1 = kernel / np.sum(kernel)

        conv_list = []
        for k in range(data.shape[2]):
            cut_data_k = data[:, :, k]
            lurred_k = signal.fftconvolve(cut_data_k, kernel1, mode="same")
            conv_list.append(lurred_k[:, :, None])

        pi = np.concatenate(conv_list, axis=2)
    else:
        pi = data
    return pi


def remove_nan(data1):
    # fits???????????????max?????s
    mask1_10 = data1 == data1
    mask1_1010 = np.where(mask1_10, 0, 1)
    label1, name1 = scipy.ndimage.label(mask1_1010)
    data_areas1 = scipy.ndimage.sum(mask1_1010, label1, np.arange(name1 + 1))
    minsize1 = 70000
    data_mask1_10 = (data_areas1 < minsize1) & (0 < data_areas1)
    small_mask1_10 = data_mask1_10[label1.ravel()].reshape(label1.shape)
    data1[small_mask1_10] = np.nanmax(data1)

    return data1


from PIL import Image, ImageDraw, ImageFont


def data_view_rectangl(col, imgs, infos=None, moji_size=100):
    """
    col: number of columns
    imgs: tensor or nparray with a shape of (?, y, x, 1) or (?, y, x, 3)
    infos: dictonary from CutTable
    """
    imgs = np.uint8(imgs[:, ::-1, :, 0]) if imgs.shape[3] == 1 else np.uint8(imgs[:, ::-1])
    row = (lambda x, y: x // y if x / y - x // y == 0.0 else x // y + 1)(imgs.shape[0], col)
    dst = Image.new("RGB", (imgs.shape[1] * col, imgs.shape[2] * row))

    # font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', moji_size)
    for i, arr in enumerate(imgs):
        img = Image.fromarray(arr)
        img = img.point(lambda x: x * 1.5)
        if infos is not None:
            draw = ImageDraw.Draw(img)
            # draw.text((10, 10), '%s'%infos['id'].tolist()[i], font=font)
            for j in range(len(infos["xmin"].tolist()[i])):
                draw.rectangle(
                    (
                        infos["xmin"].tolist()[i][j] * 300,
                        (1 - infos["ymax"].tolist()[i][j]) * 300,
                        infos["xmax"].tolist()[i][j] * 300,
                        (1 - infos["ymin"].tolist()[i][j]) * 300,
                    ),
                    width=2,
                )

        quo, rem = i // col, i % col
        dst.paste(img, (arr.shape[0] * rem, arr.shape[1] * quo))

    return dst
