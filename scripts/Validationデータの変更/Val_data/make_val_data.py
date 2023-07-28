import argparse
import json
import os
import sys

import astropy.io.fits
import astropy.wcs
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append("../")
import label_caliculator
import processing
import ring_augmentation

"""
example command:

python make_val_data.py /dataset/spitzer_data/ -r True
"""


def parse_args():
    parser = argparse.ArgumentParser(description="make validataion data for SSD")
    parser.add_argument("spitzer_path", metavar="DIR", help="spitzer_path")
    # parser.add_argument('savedir_name', metavar='DIR', help='savedir_name')
    # parser.add_argument("--each_region", "-r", action="store_true")

    return parser.parse_args()


def append_data(data, info, data_list, frame):
    if not np.isnan(data.sum()):
        data_list.append(data)
        frame.append(info)


def main(args):
    ################################################
    ## 領域ごとに作るのか、デフォルトの領域で作るのか選択 ##
    ################################################
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
    os.makedirs("/workspace/cut_val_png/", exist_ok=True)
    os.makedirs("/workspace/cut_val_png/region_val_png", exist_ok=True)

    ## choice catalogue from 'CH' or 'MWP'
    choice = "MWP"
    Ring_CATALOGUE = ring_augmentation.catalogue(choice)
    obj_sig = 1 / (2 * (np.log(2)) ** (1 / 2))

    #####################
    ## Val-Ring作成開始 ##
    #####################
    pbar = tqdm(range(len(val_l)))
    for i in pbar:
        ring_count = 0
        non_ring_count = 0

        fits_path = val_l[i]
        pbar.set_description(fits_path)
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
        GLON_center = (GLON_min + GLON_max) / 2
        GLON_new_min = GLON_center - 1.5
        GLON_new_max = GLON_center + 1.5

        Ring_catalogue = Ring_CATALOGUE.query("@GLON_new_min < GLON <= @GLON_new_max")
        label_cal = label_caliculator.label_caliculator(choice, w)
        label_cal.all_star(Ring_catalogue)

        size_list = [150, 300, 600, 900, 1200, 1800, 2500, 3000]
        fragment = 3
        savedir_name = "/workspace/cut_val_png/region_val_png/%s" % fits_path
        os.makedirs(savedir_name, exist_ok=True)
        os.makedirs(savedir_name + "/Ring", exist_ok=True)
        os.makedirs(savedir_name + "/NonRing", exist_ok=True)

        ring_data = []
        ring_row = []
        # non_ring_data = []
        for kk in range(len(size_list)):
            size = size_list[kk]

            ################### indexの計算 ###################
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
            ################### indexの計算 ###################

            cut_region_array, label = processing.cut_data(data, ind_array, cut_shape[0], obj_sig, label_cal, fits_path)
            label["id"] = [i for i in range(len(label))]

            for cut_region, row in zip(cut_region_array, label.iterrows()):
                ll = []
                if len(row["xmin"]) >= 1:
                    for la in range(len(row["xmin"])):
                        ll.append(
                            {
                                "Confidence": str(0),
                                "XMin": str(row["xmin"][la]),
                                "XMax": str(row["xmax"][la]),
                                "YMin": str(row["ymin"][la]),
                                "YMax": str(row["ymax"][la]),
                            }
                        )
                    Ring_or_NonRing = "Ring"
                    ring_count += 1
                    ring_data.append(cut_region)
                    ring_row.append(row)
                    cut_count = ring_count
                else:
                    Ring_or_NonRing = "NonRing"
                    non_ring_count += 1
                    cut_count = ring_count

                with open(f"{savedir_name}/{Ring_or_NonRing}/{Ring_or_NonRing}_{cut_count}.json", "w") as f:
                    json.dump(ll, f, indent=4)

                pil_image = Image.fromarray(np.uint8(cut_region * 255))
                pil_image.save(f"{savedir_name}/{Ring_or_NonRing}/{Ring_or_NonRing}_{cut_count}.png")
                cut_count += 1

        if len(ring_data) > 3000:
            slice = int(len(ring_data) / 1000)
        else:
            slice = 1
        processing.data_view_rectangl(25, np.array(ring_data)[::slice], pd.DataFrame(ring_row)[::slice]).save(
            f"{savedir_name}/Ring_data.png"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
