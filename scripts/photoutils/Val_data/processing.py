import copy
import json
import warnings

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from PIL import Image, ImageDraw
from scipy import signal
from torch.nn import functional as F


warnings.resetwarnings()
warnings.simplefilter("ignore")


def norm_rp(data, nan_data_dim=None):
    if nan_data_dim is not None:
        data_min = np.nanmin(nan_data_dim)
        std = np.nanstd(nan_data_dim)
        mean = np.nanmean(nan_data_dim)
    else:
        data_min = np.nanmin(data)
        std = np.nanstd(data)
        mean = np.nanmean(data)
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
    for k in range(dim):
        cut_data_k = array[:, :, k]
        if k == 2:
            cut_data_k_ = norm_rp(cut_data_k)
            gauss_list.append(cut_data_k_[:, :, None])
        else:
            nan_data = remove_peak(cut_data_k, k)
            cut_data_k_ = norm_rp(cut_data_k, nan_data)
            gauss_list.append(cut_data_k_[:, :, None])
    cut_data = np.concatenate(gauss_list, axis=2)

    return cut_data


def remove_peak(array, dim):
    data = array.copy()
    mean, median, std = sigma_clipped_stats(data, sigma=3)
    if dim == 0:
        fwhm = 1.98
    elif dim == 1:
        fwhm = 6.25
    elif dim == 2:
        fwhm = 1.66
    daofind = DAOStarFinder(fwhm=fwhm, threshold=5 * mean)
    sources = daofind(data)
    try:
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        same_shape_zero = np.zeros_like(data)
        for y, x in positions:
            same_shape_zero = cv2.circle(same_shape_zero, (int(y), int(x)), int(5), (255, 255, 255), -1)

        data[same_shape_zero == same_shape_zero.max()] = np.nan
        return data
    except:
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
    """
    データを切り取り、
    normalizeとresizeをする。
    """
    # shape_y = data.shape[0]
    # shape_x = data.shape[1]
    # data = data[int(shape_y / 4) : int(shape_y * 3 / 4), int(shape_x / 4) : int(shape_x * 3 / 4)]
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
    mask1_10 = data1 == data1
    mask1_1010 = np.where(mask1_10, 0, 1)
    label1, name1 = scipy.ndimage.label(mask1_1010)
    data_areas1 = scipy.ndimage.sum(mask1_1010, label1, np.arange(name1 + 1))
    minsize1 = 70000
    data_mask1_10 = (data_areas1 < minsize1) & (0 < data_areas1)
    small_mask1_10 = data_mask1_10[label1.ravel()].reshape(label1.shape)
    data1[small_mask1_10] = np.nanmax(data1)

    return data1


class imaging_validation:
    def __init__(self, data, ring_count, non_ring_count, obj_sig, fits_path, savedir_name, label_cal):
        self.data = data
        self.ring_count = ring_count
        self.non_ring_count = non_ring_count
        self.obj_sig = obj_sig
        self.fits_path = fits_path
        self.savedir_name = savedir_name
        self.label_cal = label_cal

    def cut_data(self, many_ind, cut_shape):
        self.cut_shape = int(cut_shape)
        Ring_data = []
        Ring_info = []

        for i in many_ind:
            self.offset_xmin = int(i[1])
            self.offset_ymin = int(i[0])
            extra_x_min = self.offset_xmin - self.cut_shape / 50
            extra_x_max = self.offset_xmin + self.cut_shape + self.cut_shape / 50
            extra_y_min = self.offset_ymin - self.cut_shape / 50
            extra_y_max = self.offset_ymin + self.cut_shape + self.cut_shape / 50
            data_c = self.data[int(extra_y_min) : int(extra_y_max), int(extra_x_min) : int(extra_x_max)].view()
            d = copy.deepcopy(data_c)
            d = conv(300, self.obj_sig, d)
            d = d[
                int(self.cut_shape / 52) : int(self.cut_shape * 51 / 52),
                int(self.cut_shape / 52) : int(self.cut_shape * 51 / 52),
            ]

            if np.isnan(d.sum()) or np.std(d[:, :, 0]) < 1e-9:
                pass
            else:
                self.cut_region = norm_res(d).astype(np.float32)

                self.label_cal.make_label(self.offset_xmin, self.offset_ymin, self.cut_shape)
                xmin_list, ymin_list, xmax_list, ymax_list, name_list = self.label_cal.check_list()
                self.info = {
                    "fits": self.fits_path,
                    "name": name_list,
                    "xmin": xmin_list,
                    "xmax": xmax_list,
                    "ymin": ymin_list,
                    "ymax": ymax_list,
                }
                Ring_or_NonRing = self.make_Validation_png()
                if Ring_or_NonRing == "Ring":
                    Ring_data.append(self.cut_region)
                    Ring_info.append(self.info)

        Ring_info = pd.DataFrame(Ring_info)
        return np.array(Ring_data), Ring_info

    def make_Validation_png(self):
        ll = []
        if len(self.info["xmin"]) >= 1:
            for la in range(len(self.info["xmin"])):
                ll.append(
                    {
                        "Confidence": str(0),
                        "XMin": str(self.info["xmin"][la]),
                        "XMax": str(self.info["xmax"][la]),
                        "YMin": str(self.info["ymin"][la]),
                        "YMax": str(self.info["ymax"][la]),
                    }
                )
            Ring_or_NonRing = "Ring"
            self.ring_count += 1
            cut_count = self.ring_count
        else:
            Ring_or_NonRing = "NonRing"
            self.non_ring_count += 1
            cut_count = self.non_ring_count

        with open(
            f"{self.savedir_name}/{Ring_or_NonRing}/{Ring_or_NonRing}_{cut_count}_{self.offset_ymin}_{self.offset_xmin}_{self.cut_shape}_{self.fits_path.split('_')[1]}_.json",
            "w",
        ) as f:
            json.dump(ll, f, indent=4)
        pil_image = Image.fromarray(np.uint8(self.cut_region * 255))
        pil_image.save(
            f"{self.savedir_name}/{Ring_or_NonRing}/{Ring_or_NonRing}_{cut_count}_{self.offset_ymin}_{self.offset_xmin}_{self.cut_shape}_{self.fits_path.split('_')[1]}_.png"
        )
        cut_count += 1

        return Ring_or_NonRing


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
