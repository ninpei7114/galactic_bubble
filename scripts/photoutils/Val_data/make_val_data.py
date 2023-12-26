import argparse
import os
import sys

import astropy.io.fits
import astropy.wcs
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../")
import label_caliculator
import processing
import select_catalogue

"""
example command:

python make_val_data.py /dataset/spitzer_data/
"""


def parse_args():
    parser = argparse.ArgumentParser(description="make validataion data for SSD")
    parser.add_argument("spitzer_path", metavar="DIR", help="spitzer_path")
    parser.add_argument("savedir_name", metavar="DIR", help="savedir_name")
    # parser.add_argument("--each_region", "-r", action="store_true")

    return parser.parse_args()


def append_data(data, info, data_list, frame):
    if not np.isnan(data.sum()):
        data_list.append(data)
        frame.append(info)


def main(args):
    """Validationデータを作成する

    Args:
        args (argparser): argparser

    example command:
    >>> python make_val_data.py /dataset/spitzer_data/
    """
    ## 各領域ごとにVal-Ringを作成する
    ## 'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため使用しない
    # fmt: off
    val_l = [
    'spitzer_00300+0000_rgb','spitzer_00600+0000_rgb','spitzer_00900+0000_rgb','spitzer_01200+0000_rgb',
    'spitzer_01500+0000_rgb','spitzer_01800+0000_rgb','spitzer_02100+0000_rgb','spitzer_02400+0000_rgb',
    'spitzer_02700+0000_rgb','spitzer_03000+0000_rgb','spitzer_03300+0000_rgb','spitzer_03600+0000_rgb',
    'spitzer_03900+0000_rgb','spitzer_04200+0000_rgb','spitzer_04500+0000_rgb','spitzer_04800+0000_rgb',
    'spitzer_05100+0000_rgb','spitzer_05400+0000_rgb','spitzer_05700+0000_rgb','spitzer_06000+0000_rgb',
    'spitzer_29700+0000_rgb','spitzer_30000+0000_rgb','spitzer_30300+0000_rgb','spitzer_30600+0000_rgb',
    'spitzer_30900+0000_rgb','spitzer_31200+0000_rgb','spitzer_31500+0000_rgb','spitzer_31800+0000_rgb',
    'spitzer_32100+0000_rgb','spitzer_32400+0000_rgb','spitzer_32700+0000_rgb','spitzer_33000+0000_rgb',
    'spitzer_33300+0000_rgb','spitzer_33600+0000_rgb','spitzer_33900+0000_rgb','spitzer_34200+0000_rgb',
    'spitzer_34500+0000_rgb','spitzer_34800+0000_rgb','spitzer_35100+0000_rgb','spitzer_35400+0000_rgb',
    'spitzer_35700+0000_rgb']
    # fmt: on
    val_l = sorted(val_l)
    os.makedirs(f"{args.savedir_name}/cut_val_png/", exist_ok=True)
    os.makedirs(f"{args.savedir_name}/cut_val_png/region_val_png", exist_ok=True)

    ## choice catalogue from 'CH' or 'MWP'
    choice = "MWP"
    Ring_CATALOGUE = select_catalogue.catalogue(choice, ring_select=True)
    obj_sig = 1 / (2 * (np.log(2)) ** (1 / 2))

    #####################
    ## Val-Ring作成開始 ##
    #####################
    pbar = tqdm(range(len(val_l)))
    for i in pbar:
        fits_path = val_l[i]
        pbar.set_description(fits_path)
        ring_count, non_ring_count = 0, 0

        savedir_name = f"{args.savedir_name}/cut_val_png/region_val_png/{fits_path}"
        os.makedirs(savedir_name, exist_ok=True)
        os.makedirs(savedir_name + "/Ring", exist_ok=True)
        os.makedirs(savedir_name + "/NonRing", exist_ok=True)

        spitzer_rfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "r.fits")[0]
        spitzer_gfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "g.fits")[0]
        spitzer_bfits = astropy.io.fits.open(args.spitzer_path + "/" + fits_path + "/" + "b.fits")[0]
        data = np.concatenate(
            [
                processing.remove_nan(spitzer_rfits.data[:, :, None]),
                processing.remove_nan(spitzer_gfits.data[:, :, None]),
                processing.remove_nan(spitzer_bfits.data[:, :, None]),
            ],
            axis=2,
        )

        w = astropy.wcs.WCS(spitzer_rfits.header)
        GLON_min, GLAT_min = w.all_pix2world(data.shape[1], 0, 0)
        GLON_max, GLAT_max = w.all_pix2world(0, data.shape[0], 0)
        Ring_catalogue = Ring_CATALOGUE.query("@GLON_min <= GLON <= @GLON_max")
        label_cal = label_caliculator.label_caliculator(choice, w)
        label_cal.all_star(Ring_catalogue)

        imaging_validation = processing.imaging_validation(
            data, ring_count, non_ring_count, obj_sig, fits_path, savedir_name, label_cal
        )

        size_list = [150, 300, 600, 900, 1200, 1800, 2400, 3000]
        fragment = 3
        all_size_ring = []
        all_size_ring_info = pd.DataFrame(columns=["fits", "name", "xmin", "xmax", "ymin", "ymax"])

        for kk in range(len(size_list)):
            size = size_list[kk]

            ################
            ## indexの計算 ##
            ################
            cut_shape = (size, size)
            slide_pix = (int(round(cut_shape[0] / fragment)), int(round(cut_shape[1] / fragment)))
            shape = data.shape
            x_num = int(shape[1] / slide_pix[1]) - 1
            y_num = int(shape[0] / slide_pix[0]) - 1
            x_idx = np.arange(cut_shape[1] / 5, slide_pix[1] * x_num, slide_pix[1])
            y_idx = np.arange(cut_shape[0] / 5, slide_pix[0] * y_num, slide_pix[0])
            x_ind, y_ind = np.meshgrid(x_idx, y_idx)

            ind_array = []
            for x, y in zip(x_ind.ravel(), y_ind.ravel()):
                ind_array.append([y, x])
            ind_array = np.array(ind_array)

            ##########################
            ## Validationデータの作成 ##
            ##########################
            ring_data, ring_info = imaging_validation.cut_data(ind_array, cut_shape[0])
            if len(ring_info) > 0:
                all_size_ring.append(ring_data)
                all_size_ring_info = pd.concat([all_size_ring_info, ring_info])

        if len(all_size_ring) > 3000:
            slice = int(len(all_size_ring) / 1000)
        else:
            slice = 1

        if len(all_size_ring) == 0:
            pass
        else:
            processing.data_view_rectangl(
                25, np.uint8(np.concatenate(all_size_ring)[::slice] * 255), pd.DataFrame(all_size_ring_info)[::slice]
            ).save(f"{savedir_name}/Ring_data.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
