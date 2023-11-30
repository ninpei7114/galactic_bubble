import copy

import numpy as np
import scipy.ndimage
import torch
from scipy import signal
from torch.nn import functional as F


def norm(data, info=None):
    ring_min = np.min(data)
    data -= ring_min

    if info is not None:
        xmin = int(float(info["xmin"][0]) * data.shape[1])
        xmax = int(float(info["xmax"][0]) * data.shape[1])
        ymin = int(float(info["ymin"][0]) * data.shape[1])
        ymax = int(float(info["ymax"][0]) * data.shape[1])
        ring_data_k = data[ymin:ymax, xmin:xmax]

        ring_mean = np.mean(ring_data_k)
        ring_std = np.std(ring_data_k)
        max_ = ring_std * 3 + ring_mean
    else:
        std = np.std(data)
        mean = np.mean(data)
        max_ = std * 3 + mean

    data[data > max_] = max_
    data /= max_
    return data


def normalize(array, info=None):
    """
    入力: (y, x, 2 or 3)
    出力: (y ,x, 2 or 3)
    """
    gauss_list = []
    dim = array.shape[2]
    for k in range(dim):
        cut_data_k = array[:, :, k]
        if info is not None:
            cut_data_k = norm(cut_data_k, info)
        else:
            cut_data_k = norm(cut_data_k)
        gauss_list.append(cut_data_k[:, :, None])
    cut_data = np.concatenate(gauss_list, axis=2)

    return cut_data


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


def norm_res(data, info=None):
    """データを切り取り、normalizeとresizeをする。

    Args:
        data (numpy array): convolutionした生データ

    Returns:
        _type_: _description_
    """
    data_ = copy.deepcopy(data)
    if info is not None:
        data_ = normalize(data_, info)
    else:
        data_ = normalize(data_)
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
